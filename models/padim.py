import pickle
from typing import Tuple, Union

from numpy import ndarray as NDArray
from torchvision import transforms

import torch
from torch import Tensor, device as Device

from models.base import BaseModel
from utils.distance import mahalanobis_multi, mahalanobis_sq
import os
from tqdm import tqdm
import time
from utils import get_performance, write_inference_result

class PaDiM(BaseModel):
    """
    The PaDiM model
    """

    def __init__(
        self,
        num_embeddings: int = 100,
        device: Union[str, Device] = "cpu",
        backbone: str = "resnet18",
        cfg: dict = None
    ):
        super(PaDiM, self).__init__(cfg.num_embeddings, device, cfg.backbone, cfg)
        self.N = 0
        self.means = torch.zeros(
            (self.num_patches, self.num_embeddings)).to(self.device)
        self.covs = torch.zeros((self.num_patches, self.num_embeddings,
                                 self.num_embeddings)).to(self.device)
    
    def train(self, cfg, dataloader):
        PARAMS_PATH = os.path.join(cfg.params_path, cfg.name)
        if not os.path.isdir(cfg.params_path): os.makedirs(cfg.params_path)
        train_time = None
        train_start = time.time()

        for batch in tqdm(dataloader):
            if isinstance(batch, tuple) or isinstance(batch, list):
                imgs = batch[0]
            self.train_one_batch(imgs)
        
        print(">> Saving params")
        params = self.get_residuals()
        try:
            with open(PARAMS_PATH, 'wb') as f:
                pickle.dump(params, f, protocol=4)
        except:
            print("saving didnt work, saving to root directory")
            with open("./default.pickle", 'wb') as f:
                pickle.dump(params, f, protocol=4)
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
            b = embeddings.size(0)
            embeddings = embeddings.reshape(
                (-1, self.num_embeddings, self.num_patches))  # b * c * (w * h) 
            #embeddings.shape[0] == b
            for i in range(self.num_patches):
                patch_embeddings = embeddings[:, :, i]  # b * c
                for j in range(b):
                    # cov computation for each embedding at batch j
                    self.covs[i, :, :] += torch.outer(
                        patch_embeddings[j, :],
                        patch_embeddings[j, :])  # c * c
                self.means[i, :] += patch_embeddings.sum(dim=0)  # c
            self.N += b  # number of images

    def get_params(self,
                   epsilon: float = 0.01) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Computes the mean vectors and covariance matrices from the
        indermediary state
        Params
        ======
            epsilon: float - coefficient for the identity matrix
        Returns
        =======
            means: Tensor - the computed mean vectors
            covs: Tensor - the computed covariance matrices
            embedding_ids: Tensor - the embedding indices
        """
        means = self.means.detach().clone()
        covs = self.covs.detach().clone()

        identity = torch.eye(self.num_embeddings).to(self.device)
        means /= self.N
        for i in range(self.num_patches):
            covs[i, :, :] -= self.N * torch.outer(means[i, :], means[i, :])
            covs[i, :, :] /= self.N - 1  # corrected covariance
            covs[i, :, :] += epsilon * identity  # constant term
        print("means.shape",means.shape)
        print("covs.shape",covs.shape)
        return means, covs, self.embedding_ids


    def _get_inv_cvars(self, covs: Tensor) -> NDArray:
        covs = covs.to("cpu")
        inv_cvars = torch.inverse(covs)
        inv_cvars = inv_cvars.to(self.device)
        return inv_cvars

    def test(self, cfg, dataloader):
        size = cfg.size

        predict_args = {}
        y_trues = []
        y_preds = []

        means, covs, emb_id = self.get_params()
        del emb_id
        inv_cvars = self._get_inv_cvars(covs)

        pbar = enumerate(tqdm(dataloader))
        file_names = []

        inf_time = None
        inf_times = []
        
        for i, test_data in pbar:
            inf_start = time.time()
            img, y_true, file_name = test_data
            res = self.predict(img, params=(means, inv_cvars), **predict_args)
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
                w = int(size[0]/4)
                h = int(size[1]/4)
                amap = res.reshape(1, 1, w, h)
                amap = amap_transform(amap)
                save_path = None
                if cfg.save_anomaly_map:
                    save_dir = os.path.join(cfg.params_path,"ano_maps")
                    if not os.path.isdir(save_dir): os.mkdir(save_dir)
                    save_path = os.path.join(cfg.params_path,"ano_maps", file_name[0])
                self.visualizer.plot_current_anomaly_map(image=img.cpu(), amap=amap.cpu(), train_or_test="test", global_step=i, save_path=save_path, maximum_as=self.cfg.max_as_score)
                
            preds = [res.max().item()]

            y_trues.extend(y_true.numpy())
            y_preds.extend(preds)
            file_names.append(file_name)
        inf_time = sum(inf_times)
        print (f'Inference time: {inf_time} secs')
        print (f'Inference time / individual: {inf_time/len(y_trues)} secs')
        # from 1 normal to 1 anomalous
        #y_trues = list(map(lambda x: 1.0 - x, y_trues))
        performance, thresholds, y_preds_man, y_preds_auc = get_performance(y_trues, y_preds, manual_threshold=cfg.decision_threshold)
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
            write_inference_result(file_names=self.file_names, y_preds=y_preds_auc, y_trues=y_trues, outf=os.path.join(cfg.params_path, "classification_result_" + str(cfg.name) + ".json"))
            if cfg.decision_threshold:
                write_inference_result(file_names=self.file_names, y_preds=y_preds_man, y_trues=y_trues, outf=os.path.join(cfg.params_path, "classification_result_" + str(cfg.name) + "_man.json"))
            
        

        return performance


    def predict(self,
                new_imgs: Tensor,
                params: Tuple[Tensor, Tensor] = None,
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
        if params is None:
            means, covs, _ = self.get_params()
            #print(covs.shape)
            inv_cvars = self._get_inv_cvars(covs)
        else:
            means, inv_cvars = params
        
        embeddings = self._embed_batch(new_imgs)
        b, c, w, h = embeddings.shape
        # not required, but need changing of testing code
        assert b == 1, f"The batch should be of size 1, got b={b}"
        embeddings = embeddings.reshape(c, w * h).permute(1, 0)
        #print(embeddings.shape)
        #print(means.shape)
        #print(inv_cvars.shape)
        

        if compare_all:
            distances = mahalanobis_multi(embeddings, means, inv_cvars)
            distances, _ = distances.min(dim=0)
        else:
            distances = mahalanobis_sq(embeddings, means, inv_cvars)
        return torch.sqrt(distances)

    def get_residuals(self) -> Tuple[int, NDArray, NDArray, NDArray, str]:
        """
        Get the intermediary data needed to stop the training and resume later
        Returns
        =======
            N: int - the number of images
            means: Tensor - the sums of embedding vectors
            covs: Tensor - the sums of the outer product of embedding vectors
            embedding_ids: Tensor - random dimensions used for size reduction
        """
        backbone = self._get_backbone()

        def detach_numpy(t: Tensor) -> NDArray:
            return t.detach().cpu().numpy()

        return (self.N, detach_numpy(self.means), detach_numpy(self.covs),
                detach_numpy(self.embedding_ids), backbone)

    @staticmethod
    def from_residuals(N: int, means: NDArray, covs: NDArray,
                       embedding_ids: NDArray, backbone: str,
                       device: Union[Device, str], cfg: dict):
        num_embeddings, = embedding_ids.shape
        padim = PaDiM(num_embeddings=num_embeddings,
                      device=device,
                      backbone=backbone,
                      cfg=cfg)
        padim.embedding_ids = torch.tensor(embedding_ids).to(device)
        padim.N = N
        padim.means = torch.tensor(means).to(device)
        padim.covs = torch.tensor(covs).to(device)

        return padim
