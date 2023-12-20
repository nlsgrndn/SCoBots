import os, sys

from model.space.postprocess_latent_variables import convert_to_boxes, retrieve_latent_repr_from_logs
from utils.checkpointer import Checkpointer
_curent_dir = os.getcwd()
for _cd in [_curent_dir, _curent_dir + "/post_eval"]:
    if _cd not in sys.path:
        sys.path.append(_cd)

import os.path as osp
import numpy as np
from engine.utils import get_config
from model import get_model
from src_utils import open_image
import os
import torch
import pandas as pd
import joblib
from dataset.atari_labels import label_list_for
from utils.bbox_matching import match_bounding_boxes_z_what, match_bounding_boxes_v2
import pickle
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
from model.space.postprocess_latent_variables import z_where_to_bb_format
from dataset.atari_labels import filter_relevant_boxes_masks
from dataset import get_dataset, get_dataloader

def create_latent_dataset(cfg):
    game = cfg.gamelist[0]
    model = get_model(cfg)
    model = model.to(cfg.device)
    checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt)
    checkpointer.load(cfg.resume_ckpt, model, None, None, cfg.device)

    dataset_mode = "test"
    dataset = get_dataset(cfg, dataset_mode)
    dataloader = get_dataloader(cfg, dataset_mode, no_shuffle_overwrite=True)

    base_path = osp.join(cfg.dataset_roots.ATARI, game, "latents", dataset_mode)
    os.makedirs(base_path, exist_ok=True)

    for i, (image_tensor, _, _, _) in enumerate(tqdm(dataloader)):
        global_step=10000000
        with torch.no_grad():
            _ , space_log = model(image_tensor, global_step=global_step)
        # (B, T, N, 4), (B, T, N,), (B, T, N), (B, T, N, 32) or (T, N, 4), (T, N,), (T, N), (T, N, 32) if batch size 1
        z_where, _, z_pres_prob, z_what = retrieve_latent_repr_from_logs(space_log)
        if len(z_where.shape) == 3:
            z_where = z_where.unsqueeze(0)
            z_pres_prob = z_pres_prob.unsqueeze(0)
            z_what = z_what.unsqueeze(0)
        for b in range(z_where.shape[0]):
            for t in range(z_where.shape[1]):
                #represent i with 5 digits
                index = i * z_where.shape[0] + b
                torch.save(z_where[b][t], osp.join(base_path, f"{index:05}_{t}_z_where.pt"))    
                torch.save(z_pres_prob[b][t], osp.join(base_path, f"{index:05}_{t}_z_pres_prob.pt"))
                torch.save(z_what[b][t], osp.join(base_path, f"{index:05}_{t}_z_what.pt"))

if __name__ == "__main__":
    cfg = get_config()
    create_latent_dataset(cfg)