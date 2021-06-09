import argparse
import logging
import os
import pickle
import sys

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

sys.path.append("../padim")

from padim.datasets import LimitedDataset, OutlierExposureDataset
from padim import PaDiMSVDD
from semmacape import TrainingDataset

logging.basicConfig(filename="logs/padeep.log", level=logging.INFO)
handler = logging.StreamHandler(sys.stdout)

root = logging.getLogger()
root.addHandler(handler)

torch.autograd.set_detect_anomaly(True)

def train(cfg):
    PARAMS_PATH = cfg.params_path
    USE_SELF_SUPERVISION = cfg.use_self_supervision
    size = tuple(map(int, cfg.size.split("x")))
    print(size)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    padeep = PaDiMSVDD(backbone="wide_resnet50",
                       device=device,
                       n_svdds=cfg.n_svdds,
                       cfg=cfg, size=size)

    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            inplace=True,
        ),
    ])

    if "semmacape" in cfg.train_folder:
        training_dataset = TrainingDataset(
            data_dir=cfg.train_folder,
            img_transforms=img_transforms,
        )
    else:
        training_dataset = ImageFolder(
            root=cfg.train_folder, # images are always normal
            transform=img_transforms,
        )

    #normal_dataset = LimitedDataset(
    #    dataset=training_dataset,
    #    limit=cfg.train_limit,
    #)
    normal_dataset = training_dataset
    if cfg.oe_folder is not None:
        train_dataset = OutlierExposureDataset(
            normal_dataset=normal_dataset,
            outlier_dataset=ImageFolder(root=cfg.oe_folder,
                                        transform=img_transforms),
            frequency=cfg.oe_frequency,
        )
    else:
        
        train_dataset = normal_dataset
        
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        num_workers=os.cpu_count(),
        shuffle=False,
    )

    train_normal_dataloader = DataLoader(
        dataset=normal_dataset,
        batch_size=16,
        num_workers=os.cpu_count(),
        shuffle=True,
    )

    if cfg.pretrain:
        root.info("Starting pretraining")
        padeep.pretrain(train_normal_dataloader, n_epochs=cfg.ae_n_epochs)
        root.info("Pretraining done")

    root.info("Starting training")
    padeep.train(
        train_dataloader,
        n_epochs=cfg.n_epochs,
        outlier_exposure=True,
        self_supervision=USE_SELF_SUPERVISION,
    )

    print(">> Saving params")
    params = padeep.get_residuals()
    with open(PARAMS_PATH, 'wb') as f:
        pickle.dump(params, f)
    print(f">> Params saved at {PARAMS_PATH}")

    return padeep
