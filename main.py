from padim.utils.visualizer import Visualizer
from torch.utils.data.dataloader import DataLoader
from padim.datasets.dataset import ImageFolderWithPaths
import argparse
import os
import pickle
import sys
import numpy as np

# import keepsake
import torch
from torchvision import transforms

sys.path.append("./")

from padim import PaDiM
from padim import PatchCore

from train import train
from test import test


def parse_args(): 
    parser = argparse.ArgumentParser(prog="PaDiM")
    
    parser.add_argument("--dataroot", required=True, type=str)
    parser.add_argument("--params_path", required=True, type=str)
    parser.add_argument("--model", default="padim", help="model to use [padim, patchcore]")
    parser.add_argument("--name", required=True, type=str, help="Name for the train run")
    parser.add_argument("--display", default=False, action="store_true")
    parser.add_argument("--num_embeddings", default=100, type=int, help="number of randomly selected embeddings (100 for r18 and 550 for WR50")
    parser.add_argument("--size",
                        default="256x256",
                        help="image size [default=256x256]")
    parser.add_argument("--backbone", default="wide_resnet50", help="[wide_resnet50, resnet50, resnet18, efficientnetb5]")
    parser.add_argument("--inference", default=False, action="store_true", help="if inference Dataset should be used (additionally)")
    parser.add_argument("--seed", type=int, help="set seed for reproducability")
    parser.add_argument("--batchsize", type=int, default=32, help="batchsize...")
    parser.add_argument("--save_anomaly_map", default=False, action="store_true", help="if the anomaly maps should be saved")
    #patchcore
    parser.add_argument("--coreset_sampling_ratio", type=float, default=0.001)
    parser.add_argument('--n_neighbors', type=int, default=3) #instead of the implementation, the most optimal neighborhood size is 3 not 9 according to the paper
    return parser.parse_args()

def seed(seed_value):
        """ Seed 

        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print("set seed to {}".format(str(seed_value)))
        
def main():
    cfg = parse_args()
    params_dict = cfg.__dict__
    seed(cfg.seed)
    params_dict["method"] = "padim"
    cfg.size = tuple(map(int, cfg.size.split("x")))
    print("Experiment started")
    
    # get datasets
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(cfg.size),
    ])
    
    
    if cfg.inference:
        inference_dataset = ImageFolderWithPaths(root=os.path.join(cfg.dataroot, "inference"), transform=img_transforms)
        inference_dataloader = DataLoader(batch_size=1, dataset=inference_dataset)
    
    training_dataset = ImageFolderWithPaths(root=os.path.join(cfg.dataroot, "train"), transform=img_transforms)
    test_dataset = ImageFolderWithPaths(root=os.path.join(cfg.dataroot, "test"), transform=img_transforms)
    
    
    train_dataloader = DataLoader(batch_size=cfg.batchsize, dataset=training_dataset)   
    test_dataloader = DataLoader(batch_size=1, dataset=test_dataset)
    if str(cfg.model).lower() == "patchcore":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = PatchCore(device=device, backbone=cfg.backbone, cfg=cfg)
        if not os.path.exists(os.path.join(cfg.params_path, cfg.name)):
            model.train(cfg=cfg, dataloader=train_dataloader)
        if cfg.inference:
            cfg.name = cfg.name + "_inference"
            model.visualizer = Visualizer(cfg)
            model.test(cfg=cfg, dataloader=inference_dataloader)
        else:
            model.test(cfg=cfg, dataloader=test_dataloader)
        
    elif str(cfg.model).lower() == "padim":
        if os.path.exists(os.path.join(cfg.params_path, cfg.name)):
            with open(os.path.join(cfg.params_path, cfg.name), "rb") as f:
                params = pickle.load(f)
            #device = "cuda" if torch.cuda.is_available() else "cpu"
            device = "cpu"
            model = PaDiM.from_residuals(*params, device=device, cfg=cfg)
        else:
            model = train(cfg, train_dataloader)
        
        if cfg.inference:
            cfg.name = cfg.name + "_inference"
            model.visualizer = Visualizer(cfg)
            test(cfg, model, inference_dataloader)
        else:
            test(cfg, model, test_dataloader)
    else:
        raise NotImplementedError("please choose a valid model ('padim', 'patchcore')")
        
if __name__ == "__main__":
    main()
