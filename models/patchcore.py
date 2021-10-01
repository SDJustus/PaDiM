import pickle
import time
from tqdm import tqdm
from utils import embeddings_concat, get_performance, write_inference_result
import os

import numpy as np
from torchvision import transforms

import torch
from torch import Tensor
import shutil
from models.base import BaseModel

from utils.kcenter_greedy import kCenterGreedy
from sklearn.random_projection import SparseRandomProjection
import faiss
import traceback


def reshape_embedding(embedding:np.ndarray):
    _, num_embeddings, _, _ = embedding.shape
    embedding = np.ascontiguousarray(np.transpose(embedding, (0,2,3,1)).reshape((-1,num_embeddings)))
    return embedding 
    


def distance_with_faiss(x, coreset, k, device):
    index = faiss.IndexFlatL2(coreset.shape[1])
    if device != "cpu":
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(coreset)
    D, _ = index.search(x, k)
    return D ** (1 / 2)


class NN():

    def __init__(self, X=None, Y=None, p=2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

class KNN(NN):

    def __init__(self, device, X=None, Y=None, k=3, p=2, ):
        self.k = k
        self.device = device
        super().__init__(X, Y, p)

    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):

        knn = distance_with_faiss(x, self.train_pts, self.k, device=self.device)
        return knn

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)    

class PatchCore(BaseModel):
    """
    The PatchCore model
    """

    def __init__(self, num_embeddings = 2, device: str = "cpu", backbone: str = "resnet18", cfg: dict = None):
        super(PatchCore, self).__init__(num_embeddings, device, backbone, cfg)
        self.N = 0
        self.embedding_list = None
        self.embedding_coreset = None
        self.all_amaps = []
        self.file_names = []

    def train(self, cfg, dataloader):
        PARAMS_PATH = os.path.join(cfg.params_path, cfg.name)
        if not os.path.isdir(cfg.params_path): os.makedirs(cfg.params_path)
        train_time = None
        train_start = time.time()
        for test_data in tqdm(dataloader):
            imgs, _, _ = test_data
            self.train_one_batch(imgs)
        try:
            self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9, random_state=cfg.seed)
            self.randomprojector.fit(self.embedding_list)
        except ValueError:
            traceback.print_exc()
            self.randomprojector = SparseRandomProjection(n_components=self.embedding_list.shape[1])
            self.randomprojector.fit(self.embedding_list)
            print("skipping Dimensionality Reduction because dimensionality is already only {}".format(str(self.embedding_list.shape[1])))
        # TODO: use faiss for all nearest neightbour and distance computations
        selector = kCenterGreedy(self.embedding_list,0, device=self.device)
        selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[], N=int(self.embedding_list.shape[0]*cfg.coreset_sampling_ratio))
        print(selected_idx)
        # selected_idx is type list
        self.embedding_coreset = self.embedding_list[selected_idx]
        
        print('initial embedding size : ', self.embedding_list.shape)
        print('final embedding size : ', self.embedding_coreset.shape)
        print(">> Saving params")
        try:
            with open(PARAMS_PATH, 'wb') as f:
                pickle.dump(self.embedding_coreset, f, protocol=4)
        except:
            print("saving didnt work, saving to root directory")
            with open("./default.pickle", 'wb') as f:
                pickle.dump(self.embedding_coreset, f, protocol=4)
        print(f">> Params saved at {PARAMS_PATH}")
        train_time = time.time() - train_start
        print (f'Train time: {train_time} secs')

    def train_one_batch(self, imgs: Tensor) -> None:
        """
        Handle only one batch, updating the internal state
        Params
        ======
            imgs: Tensor - batch tensor of size (b * c * w * h)
        """
        with torch.no_grad():
            # b * c * w * h
            embeddings = self._embed_batch(imgs.to(self.device))
            if self.embedding_list is None:
                self.embedding_list = reshape_embedding(embeddings.cpu().detach().numpy())
                
            else:
                # very memory consuming (np.vstack not reshaping)
                self.embedding_list = np.vstack((self.embedding_list, reshape_embedding(embeddings.cpu().detach().numpy())))
            
    def test(self, cfg, dataloader):
        if cfg.inference:
            cfg.name = cfg.name.split("_")[0]
        PARAMS_PATH = os.path.join(cfg.params_path, cfg.name)
        size = cfg.size

        predict_args = {}
        y_trues = []
        y_preds = []
        if not self.embedding_coreset:
            self.embedding_coreset = pickle.load(open(PARAMS_PATH, 'rb'))

        pbar = enumerate(tqdm(dataloader))

        inf_time = None
        inf_times = []
        divisor = 16 if str(self.cfg.backbone).endswith("b5") else 8
        amap_w = int(self.cfg.size[0]/divisor)
        amap_h = int(self.cfg.size[1]/divisor)
        for i, test_data in pbar:
            inf_start = time.time()
            img, y_true, file_name = test_data
            res = self.predict(img, **predict_args)
            inf_times.append(time.time()-inf_start)
            
            
                
                
                
                #amap = amap_transform(amap)
                #amap = min_max_norm(amap)
                #save_path = None
                #if cfg.save_anomaly_map:
                #    save_dir = os.path.join(cfg.params_path,"ano_maps")
                #    if not os.path.isdir(save_dir): os.mkdir(save_dir)
                #    save_path = os.path.join(save_dir, file_name[0])
                #self.visualizer.plot_current_anomaly_map(image=img.cpu(), amap=amap.cpu(), train_or_test="test", global_step=i, save_path=save_path)
            #res = res.numpy()
            N_b = res[np.argmax(res[:,0])]
            w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b))))
            amap = w*res[:,0]
            
            pred = max(amap)
            if cfg.save_anomaly_map or cfg.display:
                amap = amap.reshape(1,1,amap_w, amap_h)
                amap_transform = transforms.Compose([
                transforms.Resize(self.cfg.size),
                transforms.GaussianBlur(5)
                ])
                amap = torch.from_numpy(amap)
                amap_resized_blur = amap_transform(amap)
                save_path=None
                if self.cfg.save_anomaly_map:
                    save_dir = os.path.join(self.cfg.params_path,"ano_maps")
                    if not os.path.isdir(save_dir): os.mkdir(save_dir)
                    save_path = os.path.join(save_dir, str(pred)+ "__" + file_name[0])
                # the maximum anomaly score over inference data set... need to be adjusted -> just vor visualization purpose
                self.visualizer.plot_current_anomaly_map(image=img.cpu(), 
                                                         amap=amap_resized_blur.cpu(), 
                                                         train_or_test="test", 
                                                         global_step=i, 
                                                         save_path=save_path, 
                                                         maximum_as=self.cfg.max_as_score)
                
                
            if np.isnan(pred):
                print(pred)
                print(res)
                print(N_b)
                print(w)
                print("DDDDDDDDDDDDDDDDDDDDDDDAAAAAAAAAAAAAAAAAAAAAAAANNNNNNNNNNNNNNNNNNNNNNNNNGGGGGGGGGGGGGGGGGGGGEEEEEEEEEEEEEEEEEEERRRRRRRRRRRRRRRRR!")
            y_trues.extend(y_true.numpy())
            y_preds.append(pred)
            self.file_names.append(file_name)
        
        inf_time = sum(inf_times)
        print (f'Inference time: {inf_time} secs')
        print (f'Inference time / individual: {inf_time/len(y_trues)} secs')
        # from 1 normal to 1 anomalous
        #y_trues = list(map(lambda x: 1.0 - x, y_trues))
        performance, thresholds, y_preds_after_threshold = get_performance(y_trues, y_preds, manual_threshold=cfg.decision_threshold)
        with open(os.path.join(cfg.params_path, str(cfg.name) + str(cfg.inference)+".txt"), "w") as f:
            f.write(str(performance))
            f.close()
        self.visualizer.plot_histogram(y_trues=y_trues, y_preds=y_preds, threshold=performance["threshold"], global_step=1, save_path=os.path.join(cfg.params_path, "hist_" + str(cfg.inference)+".png"), tag="Histogram_"+str(cfg.name))
        
        self.visualizer.plot_pr_curve(y_trues=y_trues, y_preds=y_preds, thresholds=thresholds)
        self.visualizer.plot_performance(1, performance=performance)
        self.visualizer.plot_current_conf_matrix(1, performance["conf_matrix"], save_path=os.path.join(cfg.params_path, "conf_matrix" + str(cfg.inference) + ".png"))
        if cfg.decision_threshold:
            self.visualizer.plot_current_conf_matrix(2, performance["conf_matrix_man"], save_path=os.path.join(cfg.params_path, "conf_matrix_man" + str(cfg.inference) + ".png"))
            self.visualizer.plot_histogram(y_trues=y_trues, y_preds=y_preds, threshold=performance["manual_threshold"], global_step=2, save_path=os.path.join(cfg.params_path, "hist_man_" + str(cfg.inference)+".png"), tag="Histogram_"+str(cfg.name))
        self.visualizer.plot_roc_curve(y_trues=y_trues, y_preds=y_preds, global_step=1, tag="ROC_Curve", save_path=os.path.join(cfg.params_path, "roc_" + str(cfg.inference)+".png"))
        
        if cfg.inference:
            write_inference_result(file_names=self.file_names, y_preds=y_preds_after_threshold, y_trues=y_trues, outf=os.path.join(cfg.params_path, "classification_result_" + str(cfg.name) + ".json"))
            
        return performance

    def predict(self,
                new_imgs: Tensor,
                compare_all: bool = False) -> Tensor:
        """
        Computes the distance matrix for each image * patch
        Params
        ======
            imgs: Tensor - (b * W * H) tensor of images
            params: [(Tensor, Tensor)] - optional precomputed parameters
        Returns
        =======
            distances: Tensor - (c * b) array of distances
        """
        embeddings = self._embed_batch(new_imgs)
        embeddings = reshape_embedding(embeddings.cpu().detach().numpy())
        #print(self.embedding_coreset.shape)
        #print(embeddings.shape)
        knn = KNN(X=self.embedding_coreset, k=self.cfg.n_neighbors, device=self.device)
        score_patches = knn(embeddings)
        #score_patches = torch.from_numpy(score_patches)
        #print(score_patches.shape)
        # score_patches == kNN distances -> shape = (num_patches, k)
        return score_patches
 
    
    def _embed_batch(self, imgs: Tensor) -> Tensor:
        self.model.eval()
        with torch.no_grad():
            #
            feature_1, feature_2, feature_3 = self.model(imgs.to(self.device))
        #feature_1 = torch.nn.AvgPool2d(3,1,1)(feature_1)   
        feature_2 = torch.nn.AvgPool2d(3,1,1)(feature_2)
        #print("f2:", feature_2.shape)
        feature_3 = torch.nn.AvgPool2d(3,1,1)(feature_3)
        #print("f3:", feature_3.shape)
        #embeddings = embeddings_concat(feature_1, feature_2)
        #embeddings = embeddings_concat(embeddings, feature_3)
        embeddings = embeddings_concat(feature_2, feature_3)
        return embeddings
    
    
    