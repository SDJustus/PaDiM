import pickle
import time
from tqdm import tqdm
from padim.utils import embeddings_concat, get_performance, write_inference_result
import os

import numpy as np
from torchvision import transforms

import torch
from torch import Tensor
import shutil
from padim.base import BaseModel

from padim.utils.kcenter_greedy import kCenterGreedy
from sklearn.random_projection import SparseRandomProjection
import faiss


def reshape_embedding(embedding:np.ndarray):
    _, num_embeddings, _, _= embedding.shape
    embedding = embedding.reshape((-1, num_embeddings))
    return embedding
    
def distance_matrix(x, y=None, p=2):  # pairwise distance of vectors

    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, p).sum(2)

    return dist

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

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")

        dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)
        labels = torch.argmin(dist, dim=1)
        return self.train_label[labels]

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


        #dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)

        #knn = dist.topk(self.k, largest=False)
        knn = distance_with_faiss(x, self.train_pts, self.k, device=self.device)
        #print(knn[0])
        

        return knn



class PatchCore(BaseModel):
    """
    The PatchCore model
    """

    def __init__(self, num_embeddings = 2, device: str = "cpu", backbone: str = "resnet18", cfg: dict = None):
        super(PatchCore, self).__init__(num_embeddings, device, backbone, cfg)
        self.N = 0
        self.embedding_list = None
        self.embedding_coreset = None

    def train(self, cfg, dataloader):
        PARAMS_PATH = os.path.join(cfg.params_path, cfg.name)
        train_time = None
        train_start = time.time()
        for test_data in tqdm(dataloader):
            imgs, _, _ = test_data
            self.train_one_batch(imgs)
        self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9)
        self.randomprojector.fit(self.embedding_list)
        # TODO: use faiss for all nearest neightbour and distance computations
        selector = kCenterGreedy(self.embedding_list,0,0, device=self.device)
        selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[], N=int(self.embedding_list.shape[0]*cfg.coreset_sampling_ratio))
        # selected_idx is type list
        self.embedding_coreset = self.embedding_list[selected_idx]
        
        print('initial embedding size : ', self.embedding_list.shape)
        print('final embedding size : ', self.embedding_coreset.shape)
        print(">> Saving params")
        with open(PARAMS_PATH, 'wb') as f:
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
                self.embedding_list = np.array(reshape_embedding(embeddings.cpu().detach().numpy()))
                
            else:
                # very memory consuming (np.vstack not reshaping)
                self.embedding_list = np.vstack((self.embedding_list, reshape_embedding(embeddings.cpu().detach().numpy())))
            
    def test(self, cfg, dataloader):
        #cfg.name = cfg.name.split("_")[0]
        PARAMS_PATH = os.path.join(cfg.params_path, cfg.name)
        size = cfg.size

        predict_args = {}
        y_trues = []
        y_preds = []
        if not self.embedding_coreset:
            self.embedding_coreset = pickle.load(open(PARAMS_PATH, 'rb'))

        pbar = enumerate(tqdm(dataloader))
        file_names = []

        inf_time = None
        inf_times = []
        
        for i, test_data in pbar:
            inf_start = time.time()
            img, y_true, file_name = test_data
            res = self.predict(img, **predict_args)
            inf_times.append(time.time()-inf_start)
            if cfg.display:
                amap_transform = transforms.Compose([
                    transforms.Resize(size),
                    transforms.GaussianBlur(5)
                ])
                #if "efficient" in cfg.backbone:
                #    w = int(size[0]/2)
                #    h = int(size[1]/2)
                #else:
                w = int(size[0]/8)
                h = int(size[1]/8)
                amap = res[:, 0].reshape(1, 1, w, h)
                amap = amap_transform(amap)
                save_path = None
                if cfg.save_anomaly_map:
                    save_dir = os.path.join(cfg.params_path,"ano_maps")
                    if not os.path.isdir(save_dir): os.mkdir(save_dir)
                    save_path = os.path.join(cfg.params_path,"ano_maps", file_name[0])
                self.visualizer.plot_current_anomaly_map(image=img.cpu(), amap=amap.cpu(), train_or_test="test", global_step=i, save_path=save_path)
            #res = res.numpy()
            N_b = res[torch.argmax(res[:,0])]
            w = (1 - (torch.max(torch.exp(N_b))/torch.sum(torch.exp(N_b))))
            preds = w*max(res[:,0])
            if np.isnan(preds):
                print(preds)
                print(res)
                print(N_b)
                print(w)
                print("DDDDDDDDDDDDDDDDDDDDDDDAAAAAAAAAAAAAAAAAAAAAAAANNNNNNNNNNNNNNNNNNNNNNNNNGGGGGGGGGGGGGGGGGGGGEEEEEEEEEEEEEEEEEEERRRRRRRRRRRRRRRRR!")
            y_trues.extend(y_true.numpy())
            y_preds.append(preds.numpy().item())
            file_names.append(file_name)
        inf_time = sum(inf_times)
        print (f'Inference time: {inf_time} secs')
        print (f'Inference time / individual: {inf_time/len(y_trues)} secs')
        # from 1 normal to 1 anomalous
        #y_trues = list(map(lambda x: 1.0 - x, y_trues))
        performance, thresholds, y_preds_after_threshold = get_performance(y_trues, y_preds)
        with open(os.path.join(cfg.params_path, str(cfg.name) + str(cfg.inference)+".txt"), "w") as f:
            f.write(str(performance))
            f.close()
        self.visualizer.plot_histogram(y_trues=y_trues, y_preds=y_preds, threshold=performance["threshold"], global_step=1, save_path=os.path.join(cfg.params_path, str(cfg.name) + str(cfg.inference)+".csv"), tag="Histogram_"+str(cfg.name))
        self.visualizer.plot_pr_curve(y_trues=y_trues, y_preds=y_preds, thresholds=thresholds)
        self.visualizer.plot_performance(1, performance=performance)
        self.visualizer.plot_current_conf_matrix(1, performance["conf_matrix"])
        self.visualizer.plot_roc_curve(y_trues=y_trues, y_preds=y_preds, global_step=1, tag="ROC_Curve")
        
        if cfg.inference:
            write_inference_result(file_names=file_names, y_preds=y_preds_after_threshold, y_trues=y_trues, outf=os.path.join(cfg.params_path, "classification_result_" + str(cfg.name) + ".json"))
        
        

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
        #print("start prediction")
        #print("start embedding")
        embeddings = self._embed_batch(new_imgs)
        embeddings = reshape_embedding(embeddings.cpu().detach().numpy())
        #print("finished embedding")
        #b, c, w, h = embeddings.shape
        # not required, but need changing of testing code
        #assert b == 1, f"The batch should be of size 1, got b={b}"
        #embeddings = embeddings.reshape(c, w * h).permute(1, 0)
        # TODO: use faiss for all nearest neightbour and distance computations
        #print("start KNN")
        #knn = KNN(torch.from_numpy(self.embedding_coreset).to(self.device), k=self.cfg.n_neighbors)
        #score_patches = knn(torch.from_numpy(embeddings).to(self.device))[0].cpu().detach().numpy()
        knn = KNN(X=self.embedding_coreset, k=self.cfg.n_neighbors, device=self.device)
        score_patches = knn(embeddings)
        #nbrs = NearestNeighbors(n_neighbors=self.cfg.n_neighbors, algorithm='ball_tree', metric='minkowski', p=2).fit(self.embedding_coreset)
        #score_patches, _ = nbrs.kneighbors(embeddings)
        #print("score_pathes", score_patches)
        #print("end KNN")
        #anomaly_map = score_patches[:,0].reshape((28,28))
        score_patches = torch.from_numpy(score_patches)
        return score_patches
 
    
    def _embed_batch(self, imgs: Tensor) -> Tensor:
        self.model.eval()
        with torch.no_grad():
            #
            _, feature_2, feature_3 = self.model(imgs.to(self.device))
        feature_2 = torch.nn.AvgPool2d(3,1,1)(feature_2)
        feature_3 = torch.nn.AvgPool2d(3,1,1)(feature_3)
        embeddings = embeddings_concat(feature_2, feature_3)
        return embeddings
    
    