import os

from model.space.postprocess_latent_variables import retrieve_latent_repr_from_logs
from utils.checkpointer import Checkpointer

import os.path as osp
from engine.utils import get_config
from model import get_model
import os
import torch
from tqdm import tqdm
from dataset import get_dataloader
from dataset.z_what import Atari_Z_What
import time
from model.space.inference_space import WrappedSPACEforInference

def create_latent_dataset(cfg, dataset_mode = "test", model=None):
    game = cfg.gamelist[0]
    if model is None:
        model = get_model(cfg)
        model = model.to(cfg.device)
        checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt)
        checkpointer.load_last(cfg.resume_ckpt, model, None, None, cfg.device)

    model = WrappedSPACEforInference(model)

    dataset = Atari_Z_What(cfg, dataset_mode, return_keys = ["imgs"])
    dataloader = get_dataloader(cfg, dataset_mode, dataset, no_shuffle_overwrite=True)

    base_path = osp.join(cfg.dataset_roots.ATARI, game, "latents", "validation" if dataset_mode == "val" else dataset_mode)
    os.makedirs(base_path, exist_ok=True)

    z_wheres = []
    z_pres_probs = []
    z_whats = []
    global_step=10000000
    B = dataloader.batch_size
    T = dataset.T

    print("start inference")
    for i, data_dict in enumerate(tqdm(dataloader)):
        image_tensor = data_dict["imgs"]
        image_tensor = image_tensor.to(cfg.device)
        with torch.no_grad():
            #_ , space_log = model(image_tensor, global_step=global_step)
            space_log = model(image_tensor)
        # (B*T, N, 4), (B*T, N,), (B*T, N), (B*T, N, 32)
        z_where, _, z_pres_prob, z_what = retrieve_latent_repr_from_logs(space_log)

        z_where = z_where.to(cfg.device)
        z_pres_prob = z_pres_prob.to(cfg.device)
        z_what = z_what.to(cfg.device)

        z_wheres.append(z_where)
        z_pres_probs.append(z_pres_prob)
        z_whats.append(z_what)
    print("finished inference")

    print("start saving")
    infix = f"0to{T-1}" if T > 1 else "0"
    for i in range(len(z_wheres)):
        for b in range(B):
            batch_index = i * B + b
            index = b * T
            torch.save(z_wheres[i][index:index+T], osp.join(base_path, f"{batch_index:05}_{infix}_z_where.pt"))    
            torch.save(z_pres_probs[i][index:index+T], osp.join(base_path, f"{batch_index:05}_{infix}_z_pres_prob.pt"))
            torch.save(z_whats[i][index:index+T], osp.join(base_path, f"{batch_index:05}_{infix}_z_what.pt"))
    print("finished saving")

if __name__ == "__main__":
    cfg = get_config()
    create_latent_dataset(cfg, "test")