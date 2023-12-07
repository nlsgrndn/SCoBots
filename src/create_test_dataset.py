import os, sys
_curent_dir = os.getcwd()
for _cd in [_curent_dir, _curent_dir + "/post_eval"]:
    if _cd not in sys.path:
        sys.path.append(_cd)

import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
from engine.utils import get_config
from engine.train import train
from engine.eval import eval
from engine.show import show
from model import get_model
from vis import get_vislogger
from dataset import get_dataset, get_dataloader
from utils import Checkpointer, open_image, show_image, save_image, \
    corners_to_wh, draw_bounding_boxes, place_labels
import os
from torch import nn
from torch.utils.data import Subset, DataLoader
from utils import draw_bounding_boxes
from torchvision.transforms.functional import crop
import torch
from eval.ap import read_boxes, convert_to_boxes, compute_ap, compute_counts
from tqdm import tqdm
from termcolor import colored
import pandas as pd
from PIL import Image
import PIL
import joblib
from dataset.labels import label_list_for

def get_labels(classifier, z_what):
    return classifier.predict(z_what)

def retrieve_latent_repr_from_logs(logs):
    z_where, z_pres_prob, z_what = logs['z_where'], logs['z_pres_prob'], logs['z_what']
    z_where = z_where.detach().cpu()
    z_pres_prob = z_pres_prob.detach().cpu()
    z_what = z_what.detach().cpu()
    z_pres = z_pres_prob > 0.5
    return z_where, z_pres, z_pres_prob, z_what,

# load model and configurations

print("Implicitlys sets batch dimension to 1!!!!!!!!")
cfg, task = get_config()
game = cfg.gamelist[0]
label_list = label_list_for(game)
model = get_model(cfg)
model = model.to(cfg.device)
checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt)
checkpoint = checkpointer.load(cfg.resume_ckpt, model, None, None, cfg.device)
z_classifier_path =  cfg.z_what_classifier_path
z_what_classifier = joblib.load(z_classifier_path)

folder = "test"
path_with_csvs = os.path.join(cfg.dataset_roots.ATARI, game, "bb", folder)
path_with_images = os.path.join(cfg.dataset_roots.ATARI, game, "space_like", folder)
csv_files = sorted(os.listdir(path_with_csvs))
image_files = sorted(os.listdir(path_with_images))

actual_bbs = []
predicted_bbs = []
# Iterate over the files and open/process them
for i, (csv_file, image_file) in enumerate(zip(csv_files, image_files)): # Note: all images are processed one by one
    csv_path = os.path.join(path_with_csvs, csv_file)
    image_path = os.path.join(path_with_images, image_file)
    table = pd.read_csv(csv_path, header=None, index_col=None)
    # replace label in column 5 with label index
    table[5] = table[5].apply(lambda x: label_list.index(x))
    curr_actual_bbs = []
    # remove 4th column (confidence)
    table = table.drop(columns=[4])
    # convert to numpy array
    table = table.to_numpy()
    actual_bbs.append(table)

    image = open_image(image_path).to(cfg.device)
    with torch.no_grad():
        _ , space_log = model(image, global_step=100000000)
    
    # (B, N, 4), (B, N, 1), (B, N, 1), (B, N, 32)
    z_where, z_pres, z_pres_prob, z_what = retrieve_latent_repr_from_logs(space_log)
    z_what_pres = z_what*z_pres
    z_what_pres =z_what_pres[z_what_pres!=0]
    z_what_pres = z_what_pres.view(-1, 32)

    # retrieve labels and bounding
    from dataset.labels import filter_relevant_boxes_masks
    #assuming batch size is 1 (therefore [0])
    boxes_batch = convert_to_boxes(z_where, z_pres.squeeze(-1), z_pres_prob.squeeze(-1))
    mask = filter_relevant_boxes_masks(game, boxes_batch, None)[0]
    boxes_batch = boxes_batch[0]
    # find out the indices of the boxes that are 
    labels = get_labels(z_what_classifier, z_what_pres)
    labels = labels[:, None]
    boxes_and_labels = np.concatenate((boxes_batch, labels), axis=1)
    print(boxes_and_labels.shape)
    boxes_and_labels = boxes_and_labels[mask]
    print(boxes_and_labels.shape)
    predicted_bbs.append(boxes_and_labels)

    if i >= 31:
        break

if not os.path.exists(f"labeled/{cfg.exp_name}"):
    os.makedirs(f"labeled/{cfg.exp_name}")
# store as numpy files
np.savez(f"labeled/{cfg.exp_name}/actual_bbs_{folder}.npz", *actual_bbs)
np.savez(f"labeled/{cfg.exp_name}/predicted_bbs_{folder}.npz", *predicted_bbs)
print(f"Extracted z_whats and saved it in labeled/{cfg.exp_name}/")