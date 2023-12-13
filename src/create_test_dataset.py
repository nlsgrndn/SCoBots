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

def get_label_predictions(classifier, z_what):
    return classifier.predict(z_what)

def save_relevant_objects_as_images(global_step, image_refs, image, z_where_pres, image_file_name, labels):
    path = f'labeled/{cfg.exp_name}/objects/'
    if not os.path.exists(path):
        os.makedirs(path)
    for obj_idx, (bb, label) in enumerate(zip(z_where_pres, labels)):
        width, height, center_x, center_y = bb.tolist()
        y_min, y_max, x_min, x_max = z_where_to_bb_format(width, height, center_x, center_y)
        bb = np.array([x_min, y_min, x_max, y_max]) * 128 # (left, upper, right, lower)
        bb = tuple(bb.astype(int))
        new_image_path = f'gs{global_step:06}_{image_file_name.split(".")[0]}_obj{obj_idx}_label_{label}.png'
        try:
            cropped = image.crop(bb)
            cropped.save(path + new_image_path)
        except Exception as e:
            print(e)
            image.save(path + new_image_path)
        image_refs.append(new_image_path)
    return image_refs

def save_relevant_objects_as_pickle(self, global_step, labels_relevant_idx, z_whats_relevant, labels_relevant, image_refs):
    image_refs_relevant = [image_refs[idx] for idx, yes in enumerate(labels_relevant_idx) if yes]
    # Save relevant objects data to pickle file
    with open(f'{self.relevant_object_hover_path}/relevant_objects_{global_step:06}.pkl', 'wb') as output_file:
        relevant_objects_data = {
            'z_what': z_whats_relevant,
            'labels': labels_relevant,
            'image_refs': image_refs_relevant,
        }
        pickle.dump(relevant_objects_data, output_file, pickle.DEFAULT_PROTOCOL)

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
image_refs = []
# Iterate over the files and open/process them
for i, (csv_file_name, image_file_name) in enumerate(zip(csv_files, image_files)): # Note: all images are processed one by one
    csv_path = os.path.join(path_with_csvs, csv_file_name)
    image_path = os.path.join(path_with_images, image_file_name)
    table = pd.read_csv(csv_path, header=None, index_col=None)
    # replace label in column 5 with label index
    table[5] = table[5].apply(lambda x: label_list.index(x))
    curr_actual_bbs = []
    # remove 4th column (confidence)
    table = table.drop(columns=[4])
    # convert to numpy array
    table = table.to_numpy()
    actual_bbs_and_label.append(table)
    image = Image.open(image_path)
    image_tensor = TF.to_tensor(image).unsqueeze_(0).to(cfg.device)
    global_step=10000000
    with torch.no_grad():
        _ , space_log = model(image_tensor, global_step=global_step)
    
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

    z_where_pres = z_where[z_pres]
    gt_labels_for_predicted_boxes = match_bounding_boxes_v2(table, boxes_batch)
    save_relevant_objects_as_images(global_step, image_refs, image, z_where_pres, image_file_name, gt_labels_for_predicted_boxes)

    if i >= 128:
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
from dataset.atari_labels import get_moving_indices
relevant_indices = get_moving_indices(game)


mask = np.isin(labels_collector, relevant_indices)
labels_collector_relevant = labels_collector[mask]
z_whats_collector_relevant =z_whats_collector[mask]

if not os.path.exists(f"labeled/{cfg.exp_name}"):
    os.makedirs(f"labeled/{cfg.exp_name}")
# store as numpy files
np.savez(f"labeled/{cfg.exp_name}/actual_bbs_and_label_{folder}.npz", *actual_bbs_and_label)
np.savez(f"labeled/{cfg.exp_name}/predicted_bbs_and_label_{folder}.npz", *predicted_bbs_and_label)
np.savez(f"labeled/{cfg.exp_name}/predicted_bbs_and_z_what_{folder}.npz", *predicted_bbs_and_z_what)
# save image refs as csv
df = pd.DataFrame(image_refs)
df.to_csv(f"labeled/{cfg.exp_name}/image_refs_all_{folder}.csv", index=False)
np.save(f"labeled/{cfg.exp_name}/labels_relevant_{folder}.npy", labels_collector_relevant)
np.save(f"labeled/{cfg.exp_name}/z_whats_relevant_{folder}.npy", z_whats_collector_relevant)
np.save(f"labeled/{cfg.exp_name}/labels_all_{folder}.npy", labels_collector)
np.save(f"labeled/{cfg.exp_name}/z_whats_all_{folder}.npy", z_whats_collector)
print(f"Extracted z_whats and saved it in labeled/{cfg.exp_name}/")