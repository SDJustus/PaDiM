import pickle
import time
import torch
from tqdm import tqdm
from padim import PaDiM
import os



def train(cfg, dataloader):
    PARAMS_PATH = os.path.join(cfg.params_path, cfg.name)
    if not os.path.isdir(cfg.params_path): os.makedirs(cfg.params_path)
    train_time = None
    train_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    padim = PaDiM(num_embeddings=cfg.num_embeddings, device=device, backbone=cfg.backbone, cfg=cfg)

    for batch in tqdm(dataloader):
        if isinstance(batch, tuple) or isinstance(batch, list):
            imgs = batch[0]
        padim.train_one_batch(imgs)
    
    print(">> Saving params")
    params = padim.get_residuals()
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

    return padim
