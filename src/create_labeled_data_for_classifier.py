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
    corners_to_wh, draw_bounding_boxes, get_labels, place_labels
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

def process_image(log, image_rgb, idx):
    # (B, N, 4), (B, N, 1), (B, N, D) B = batch size, N = number of cells in the grid, D = dimension of the latent space
    z_where, z_pres_prob, z_what = log['z_where'], log['z_pres_prob'], log['z_what']
    print(z_where.shape, z_pres_prob.shape, z_what.shape)
    # (B, N, 4), (B, N), (B, N)
    z_where = z_where.detach().cpu()
    z_pres_prob = z_pres_prob.detach().cpu().squeeze()
    z_pres = z_pres_prob > 0.0002
    z_what_pres = (z_what * z_pres.unsqueeze(-1)).view(4, -1)
    print(z_where.shape, z_pres_prob.shape, z_what_pres.shape)
    boxes_batch = np.array(convert_to_boxes(z_where, z_pres.unsqueeze(0), z_pres_prob.unsqueeze(0))).squeeze()
    labels = get_labels(table.iloc[[idx]], boxes_batch)
    assert z_what_pres.shape[0] == labels.shape[0]
    if action == "visu":
        visu = place_labels(labels, boxes_batch, image_rgb)
        visu = draw_bounding_boxes(visu, boxes_batch, labels)
        show_image(visu)
        exit()
        # show_image(image_fs[0])
    all_z_what.append(z_what_pres.detach().cpu())
    all_labels.append(labels.detach().cpu())


def img_path_to_tensor(path):
    pil_img = Image.open(path).convert('RGB')
    pil_img = pil_img.resize((128, 128), PIL.Image.BILINEAR)
    image = np.array(pil_img)
    return torch.from_numpy(image / 255).permute(2, 0, 1).float()

folder = "test"

action = ["visu", "extract"][1]

folder_sizes = {"train": 50000, "test": 500, "validation": 500}
nb_images = folder_sizes[folder]
cfg, task = get_config()

TIME_CONSISTENCY = True
#EXTRACT_IMAGES = False
USE_FULL_SIZE = True


# load model and configurations
game = cfg.gamelist[0]
model = get_model(cfg)
model = model.to(cfg.device)
checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name),
                            max_num=cfg.train.max_ckpt, load_time_consistency=TIME_CONSISTENCY)
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
        loss, space_log = model.space(image, global_step=100000000)
    
    # (B, N, 4), (B, N, 1), (B, N, 32)
    z_where, z_pres_prob, z_what = space_log['z_where'], space_log['z_pres_prob'], space_log['z_what']
    z_where = z_where.detach().cpu()
    z_pres_prob = z_pres_prob.detach().cpu()
    z_pres = z_pres_prob > 0.5
    z_what_pres = z_what*z_pres
    z_what_pres =z_what_pres[z_what_pres!=0]
    # specify that z_what_pres should have 32 in the last dimension
    z_what_pres = z_what_pres.view(-1, 32)
    print(z_what_pres.shape)


    # retrieve labels and bounding 
    boxes_batch = convert_to_boxes(z_where, z_pres.squeeze(-1), z_pres_prob.squeeze(-1))
    labels = table.iloc[:,5]
    if labels.shape[0] != boxes_batch[0].shape[0]:
        print(colored(f"Warning: {labels.shape[0]} labels and {boxes_batch[0].shape[0]} boxes", "red"))
        
    
    # transform labels to tensor by mapping label strings to integer values
    from dataset.labels import label_list_pong

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
    

    



#for i in tqdm(range(0, nb_images, 4 if TIME_CONSISTENCY else 1)):
#    print(i)
#    if TIME_CONSISTENCY:
#        fn = [f"../aiml_atari_data/{game}/space_like/{folder}/{i:05}_{j}.png" for j in range(4)]
#        image = torch.stack([img_path_to_tensor(f) for f in fn]).to(cfg.device).unsqueeze(0)
#        print(image.shape)
#        image = image.squeeze(0)
#        print(image.shape)
#        img_path_fs = [f"../aiml_atari_data/{game}/rgb/{folder}/{i:05}_{j}.png" for j in range(4)]
#        image_fs = torch.stack([img_path_to_tensor(f) for f in img_path_fs]).to(cfg.device).unsqueeze(0)
#    else:
#        img_path = f"../aiml_atari_data/{game}/space_like/{folder}/{i:05}.png"
#        image = open_image(img_path).to(cfg.device)
#        img_path_fs = f"../aiml_atari_data/{game}/rgb/{folder}/{i:05}.png"
#        image_fs = open_image(img_path_fs).to(cfg.device)
#
#    # TODO: treat global_step in a more elegant way
#    with torch.no_grad():
#        loss, space_log = model.space(image, global_step=100000000)
#    if TIME_CONSISTENCY:
#        for j in range(4):
#            process_image(space_log, image_fs, i + j)
#    else:
#        process_image(space_log, image_fs, i)

all_z_what = torch.cat(all_z_what)
all_labels = torch.cat(all_labels)

if action == "extract":
    if not os.path.exists(f"labeled/{cfg.exp_name}"):
        os.makedirs(f"labeled/{cfg.exp_name}")
    torch.save(all_z_what, f"labeled/{cfg.exp_name}/z_what_{folder}.pt")
    torch.save(all_labels, f"labeled/{cfg.exp_name}/labels_{folder}.pt")
    print(f"Extracted z_whats and saved it in labeled/{cfg.exp_name}/")

# import ipdb; ipdb.set_trace()
# image = (image[0] * 255).round().to(torch.uint8)  # for draw_bounding_boxes
# image_fs = (image_fs[0] * 255).round().to(torch.uint8)  # for draw_bounding_boxes

# image_fs = place_labels(labels, boxes_batch, image_fs)


# if USE_FULL_SIZE:
#     show_image(draw_bounding_boxes(image_fs, boxes_batch))
# else:
#     show_image(draw_bounding_boxes(image, boxes_batch))

# exit()
# if EXTRACT_IMAGES:
#     for j, bbox in enumerate(torch.tensor(bb)):
#         top, left, height, width = corners_to_wh(bbox)
#         cropped = crop(image.to("cpu"), top, left, height, width)
#         # show_image(cropped)
#         save_image(cropped, img_path, j)




