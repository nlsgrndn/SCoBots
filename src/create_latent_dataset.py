import os

from model.space.postprocess_latent_variables import retrieve_latent_repr_from_logs
from utils.checkpointer import Checkpointer

import os.path as osp
from engine.utils import get_config
from model import get_model
import os
import torch
from tqdm import tqdm
from dataset import get_dataset, get_dataloader

def create_latent_dataset(cfg, dataset_mode="test"):
    game = cfg.gamelist[0]
    model = get_model(cfg)
    model = model.to(cfg.device)
    checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt)
    checkpointer.load(cfg.resume_ckpt, model, None, None, cfg.device)

    dataset = get_dataset(cfg, dataset_mode)
    dataloader = get_dataloader(cfg, dataset_mode, no_shuffle_overwrite=True)

    base_path = osp.join(cfg.dataset_roots.ATARI, game, "latents", dataset_mode)
    os.makedirs(base_path, exist_ok=True)

    #ASSUMES BATCH SIZE 1, designed to work with cpu #TODO adapt to gpu
    for i, (image_tensor, _, _, _) in enumerate(tqdm(dataloader)):
        global_step=10000000
        image_tensor = image_tensor.to(cfg.device)
        #further splitup dimension 1 (T) into individual frames
        for t in tqdm(range(image_tensor.shape[1])):
            with torch.no_grad():
                _ , space_log = model(image_tensor[:, t:t+1, ...], global_step=global_step) # model requires 5D or 4D input, in this case we give it 5D input (with first two dimensions being 1)
            
            # (B, T, N, 4), (B, T, N,), (B, T, N), (B, T, N, 32) or (1, N, 4), (1, N,), (1, N), (1, N, 32) because batch size is 1 and one cons. frame is indexed at a time
            z_where, _, z_pres_prob, z_what = retrieve_latent_repr_from_logs(space_log)
            z_where = z_where.to(cfg.device)
            z_pres_prob = z_pres_prob.to(cfg.device)
            z_what = z_what.to(cfg.device)

            torch.save(z_where[0], osp.join(base_path, f"{i:05}_{t}_z_where.pt"))    
            torch.save(z_pres_prob[0], osp.join(base_path, f"{i:05}_{t}_z_pres_prob.pt"))
            torch.save(z_what[0], osp.join(base_path, f"{i:05}_{t}_z_what.pt"))

if __name__ == "__main__":
    cfg = get_config()
    create_latent_dataset(cfg)