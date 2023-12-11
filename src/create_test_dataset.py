import os, sys

from model.space.postprocess_latent_variables import convert_to_boxes, retrieve_latent_repr_from_logs
from checkpointer import Checkpointer
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
from bbox_matching import match_bounding_boxes_z_what

def get_label_predictions(classifier, z_what):
    return classifier.predict(z_what)

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

actual_bbs_and_label = []
predicted_bbs_and_label = []
predicted_bbs_and_z_what = []
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
    actual_bbs_and_label.append(table)

    image = open_image(image_path).to(cfg.device)
    with torch.no_grad():
        _ , space_log = model(image, global_step=100000000)
    
    # (B, N, 4), (B, N,), (B, N,), (B, N, 32)
    z_where, z_pres, z_pres_prob, z_what = retrieve_latent_repr_from_logs(space_log)
    z_what_pres = z_what[z_pres]

    # retrieve labels and bounding
    #assuming batch size is 1 (therefore [0])
    boxes_batch = convert_to_boxes(z_where, z_pres, z_pres_prob)[0] #TODO: maybe use filter_relevant_boxes_masks instead


    boxes_and_z_what_pres = np.concatenate((boxes_batch, z_what_pres), axis=1)
    predicted_bbs_and_z_what.append(boxes_and_z_what_pres)

    labels = get_label_predictions(z_what_classifier, z_what_pres)
    labels = labels[:, None]
    boxes_and_labels = np.concatenate((boxes_batch, labels), axis=1)

    predicted_bbs_and_label.append(boxes_and_labels)

    if i >= 127:
        break

labels_collector = []
z_whats_collector = []
for i in range(len(actual_bbs_and_label)):
    gt_labels, z_what_embeddings = match_bounding_boxes_z_what(actual_bbs_and_label[i], predicted_bbs_and_z_what[i])
    labels_collector.extend(gt_labels)
    z_whats_collector.extend(z_what_embeddings)
labels_collector = np.array(labels_collector)
z_whats_collector = np.array(z_whats_collector)


# filter irrelevant boxes
relevant_indices_for_game = {"pong": [1,2,4], "boxing": [2, 4]}
relevant_indices = relevant_indices_for_game[cfg.exp_name]


mask = np.isin(labels_collector, relevant_indices)
labels_collector_relevant = labels_collector[mask]
z_whats_collector_relevant =z_whats_collector[mask]

if not os.path.exists(f"labeled/{cfg.exp_name}"):
    os.makedirs(f"labeled/{cfg.exp_name}")
# store as numpy files
np.savez(f"labeled/{cfg.exp_name}/actual_bbs_and_label_{folder}.npz", *actual_bbs_and_label)
np.savez(f"labeled/{cfg.exp_name}/predicted_bbs_and_label_{folder}.npz", *predicted_bbs_and_label)
np.savez(f"labeled/{cfg.exp_name}/predicted_bbs_and_z_what_{folder}.npz", *predicted_bbs_and_z_what)
np.save(f"labeled/{cfg.exp_name}/labels_relevant_{folder}.npy", labels_collector_relevant)
np.save(f"labeled/{cfg.exp_name}/z_whats_relevant_{folder}.npy", z_whats_collector_relevant)
np.save(f"labeled/{cfg.exp_name}/labels_all_{folder}.npy", labels_collector)
np.save(f"labeled/{cfg.exp_name}/z_whats_all_{folder}.npy", z_whats_collector)
print(f"Extracted z_whats and saved it in labeled/{cfg.exp_name}/")