import os, sys

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
from termcolor import colored
import pandas as pd
from model.space.postprocess_latent_variables import convert_to_boxes, retrieve_latent_repr_from_logs

folder = "test"

action = ["visu", "extract"][1]

folder_sizes = {"train": 50000, "test": 500, "validation": 500}
nb_images = folder_sizes[folder]
cfg, task = get_config()

TIME_CONSISTENCY = True
#EXTRACT_IMAGES = False
#USE_FULL_SIZE = True


# load model and configurations
game = cfg.gamelist[0]
model = get_model(cfg)
model = model.to(cfg.device)
checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt,)
if cfg.resume:
    checkpoint = checkpointer.load(cfg.resume_ckpt, model, None, None, cfg.device)

path_with_csvs = os.path.join(cfg.dataset_roots.ATARI, game, "bb", folder)
path_with_images = os.path.join(cfg.dataset_roots.ATARI, game, "space_like", folder)
csv_files = sorted(os.listdir(path_with_csvs))
image_files = sorted(os.listdir(path_with_images))

all_z_what = []
all_labels = []
# Iterate over the files and open/process them
for i, (csv_file, image_file) in enumerate(zip(csv_files, image_files)): # Note: all images are processed one by one
    print(colored(f"Processing {csv_file} and {image_file}", "green"))
    csv_path = os.path.join(path_with_csvs, csv_file)
    image_path = os.path.join(path_with_images, image_file)
    # read csv without header or index
    table = pd.read_csv(csv_path, header=None, index_col=None)
    image = open_image(image_path).to(cfg.device)
    with torch.no_grad():
        loss, space_log = model(image, global_step=100000000)
    
    # (B, N, 4), (B, N,), (B, N,), (B, N, 32)
    z_where, z_pres, z_pres_prob, z_what = retrieve_latent_repr_from_logs(space_log)
    z_what_pres = z_what[z_pres]

    # retrieve labels and bounding 
    boxes_batch = convert_to_boxes(z_where, z_pres, z_pres_prob)
    labels = table.iloc[:,5]
    if labels.shape[0] != boxes_batch[0].shape[0]:
        print(colored(f"Warning: {labels.shape[0]} labels and {boxes_batch[0].shape[0]} boxes", "red"))
        
    
    # transform labels to tensor by mapping label strings to integer values
    from dataset.atari_labels import label_list_pong

    # assign ground truth labels to detected bounding boxes by using the distance to the RAM bounding boxes
    labels = []
    for j in range(boxes_batch[0].shape[0]):
        cur_bbox = boxes_batch[0][j]
        min_dist = np.inf
        label = ""
        cor_pos = None
        #iterate over rows in dataframe
        for idx, row in table.iterrows():
            name = row[5]
            pos_x = (row[2] + row[3]) / 2
            pos_y = (row[0] + row[1]) / 2
            cur_bbox_x = (cur_bbox[2] + cur_bbox[3]) / 2
            cur_bbox_y = (cur_bbox[0] + cur_bbox[1]) / 2
            cur_dist = np.sqrt((pos_x - cur_bbox_x)**2 + (pos_y - cur_bbox_y)**2)
            if cur_dist < min_dist:
                label = name
                min_dist = cur_dist
                cor_pos = (pos_x, pos_y)
        labels.append(label)


    pong_label2int_mapping = {l: i for i, l in enumerate(label_list_pong)}
    labels = torch.LongTensor([pong_label2int_mapping[l] for l in labels])
    # store the data
    all_z_what.append(z_what_pres.detach().cpu())
    all_labels.append(labels.detach().cpu())

    if i >= 31:
        break

all_z_what = torch.cat(all_z_what)
all_labels = torch.cat(all_labels)

if action == "extract":
    if not os.path.exists(f"labeled/{cfg.exp_name}"):
        os.makedirs(f"labeled/{cfg.exp_name}")
    torch.save(all_z_what, f"labeled/{cfg.exp_name}/z_what_{folder}.pt")
    torch.save(all_labels, f"labeled/{cfg.exp_name}/labels_{folder}.pt")
    print(f"Extracted z_whats and saved it in labeled/{cfg.exp_name}/")




