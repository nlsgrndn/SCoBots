import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
import skvideo.io as skv
from PIL import Image
import PIL
import os.path as osp
from .atari_labels import to_relevant, filter_relevant_boxes
import pandas as pd
import cv2 as cv
from skimage.morphology import (disk, square)
from skimage.morphology import (erosion, dilation, opening, closing, white_tophat, skeletonize)
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes as draw_bb
from utils.bbox_matching import match_bbs, match_bounding_boxes_v2
from dataset.atari_labels import label_list_for, no_label_str, filter_relevant_boxes_masks, get_moving_indices
from model.space.postprocess_latent_variables import convert_to_boxes, retrieve_latent_repr_from_logs


#class Atari_All(Dataset):
#    def __init__(self, cfg, dataset_mode):
#        self.game = cfg.gamelist[0]
#
#        dataset_mode = 'validation' if dataset_mode == 'val' else dataset_mode
#        self.dataset_mode = dataset_mode
#
#        self.label_subset_type = "all" # "all", "moving", "relevant
#        self.T = 128 # number of consecutive frames per sample
#        self.labels = label_list_for(cfg.gamelist[0])
#
#        self.motion_kind = cfg.arch.motion_kind # "mode" or "flow"
#
#        # folder paths
#        img_folder = "space_like"
#        self.image_base_path = osp.join(cfg.dataset_roots.ATARI, self.game, img_folder)
#        self.motion_base_path = osp.join(cfg.dataset_roots.ATARI, self.game, self.motion_kind)
#        self.bb_base_path = osp.join(cfg.dataset_roots.ATARI, self.game, "bb")
#        self.latent_base_path = osp.join(cfg.dataset_roots.ATARI, self.game, "latents")
#
#
#        self.image_fn_count = len([True for img in os.listdir(self.image_path) if img.endswith(".png")])
#
#
#
#    def __getitem__(self, stack_idx): # (T, ...) where T is number of consecutive frames and ... represents the actual dimensions of the data
#        imgs = torch.stack([transforms.ToTensor()(self.read_img(stack_idx, i)) for i in range(self.T)])
#        gt_bbs = self.get_gt_bbs(stack_idx * self.T, (stack_idx + 1) * self.T)
#        gt_labels = self.get_labels(gt_bbs, imgs)
#        predicted_bbs = torch.stack([self.read_tensor(stack_idx, i) for i in range(self.T)])
#        z_whats_of_predicted_bbs = torch.stack([self.read_tensor(stack_idx, i, "z_what") for i in range(self.T)])
#        # problem: 3rd dimension of gt_bbs, gt_labels, predicted_bbs, z_whats_of_predicted_bbs can all vary internally -> Tensors cannot be stacked
#        # solution: maybe pad with nan or something
#        return imgs, gt_bbs, gt_labels, predicted_bbs, z_whats_of_predicted_bbs
#
#    def __len__(self):
#        return self.image_fn_count // self.T
#
#    def read_img(self, stack_idx, i):
#        path = os.path.join(self.image_base_path, self.dataset_mode, f'{stack_idx:05}_{i}.png')
#        return np.array(Image.open(path).convert('RGB'))
#
#    def read_tensor(self, stack_idx, i, postfix=None):
#        path = os.path.join(self.motion_path,
#                            f'{stack_idx:05}_{i}_{postfix}.pt'
#                            if postfix else f'{stack_idx:05}_{i}.pt')
#        return torch.load(path)
    
class Atari_Z_What(Dataset):
    def __init__(self, cfg, dataset_mode, boxes_subset="all", return_keys=None):
        self.game = cfg.gamelist[0]

        self.return_keys = return_keys

        #self.gt_boxes_subset = #"all" # "all", "moving"
        #self.label_subset_type = #"all+no_label" # "moving+no_label", "moving"
        #self.filter_predicted_bbs = #False # True, False

        dataset_mode = 'validation' if dataset_mode == 'val' else dataset_mode
        self.dataset_mode = dataset_mode

        self.boxes_subset = boxes_subset # "all", "relevant"
        self.T = 4 # number of consecutive frames per sample
        self.labels = label_list_for(cfg.gamelist[0])

        # folder paths
        img_folder = "space_like"
        self.image_base_path = osp.join(cfg.dataset_roots.ATARI, self.game, img_folder)
        self.bb_base_path = osp.join(cfg.dataset_roots.ATARI, self.game, "bb")
        self.latents_base_path = osp.join(cfg.dataset_roots.ATARI, self.game, "latents")

        self.image_fn_count = len([True for img in os.listdir(self.image_path) if img.endswith(".png")])

    def __getitem__(self, stack_idx): # (T, ...) where T is number of consecutive frames and ... represents the actual dimensions of the data
        
        base_path = self.image_path
        imgs = torch.stack([transforms.ToTensor()(self.read_img(stack_idx, i, base_path)) for i in range(self.T)])
        
        base_path = self.latents_path
        z_whats = torch.stack([self.read_tensor(stack_idx, i, base_path, "z_what") for i in range(self.T)])
        z_pres_probs = torch.stack([self.read_tensor(stack_idx, i, base_path, "z_pres_prob") for i in range(self.T)])
        z_wheres = torch.stack([self.read_tensor(stack_idx, i, base_path, "z_where") for i in range(self.T)])
        z_pres_s = z_pres_probs > 0.5

        base_path = self.bb_path
        gt_bbs_and_labels = [self.read_csv(stack_idx, i, base_path) for i in range(self.T)] # can't be stacked because of varying number of objects
        
        if self.boxes_subset == "moving": # modify labels of static objects to "no_label"
            for i in range(self.T):
                mask = gt_bbs_and_labels[i][:, 4] == 0 # static objects
                index_for_no_label = label_list_for(self.game).index(no_label_str)
                gt_bbs_and_labels[i][mask, 5] = index_for_no_label

        if self.boxes_subset == "relevant":  # remove non-moving gt boxes
            gt_bbs_and_labels = [gt_bbs_and_labels[i][gt_bbs_and_labels[i][:, 4] == 1] for i in range(self.T)]

        pred_boxes = convert_to_boxes(z_wheres, z_pres_s, z_pres_probs, with_conf=True) # list of tensors of shape (N, 4) where N is number of objects in that frame

        z_whats_pres_s = []
        for i in range(self.T):
            z_whats_pres_s.append(z_whats[i][z_pres_s[i]])
        
        gt_labels_for_pred_boxes = []
        for i in range(self.T):
            #convert type of tensor from float to int: use method 
            gt_labels_for_pred_boxes.append(match_bounding_boxes_v2(torch.Tensor(gt_bbs_and_labels[i]), torch.Tensor(pred_boxes[i])).reshape(-1, 1).to(torch.int)) 

        if self.boxes_subset == "relevant":
            masks = filter_relevant_boxes_masks(self.game, pred_boxes, None)
            for i in range(self.T):
                pred_boxes[i] = pred_boxes[i][masks[i]]
                gt_labels_for_pred_boxes[i] = gt_labels_for_pred_boxes[i][masks[i]]
                z_whats_pres_s[i] = z_whats_pres_s[i][masks[i]]

        data = {
            "imgs": imgs,
            "z_whats": z_whats,
            "z_pres_probs": z_pres_probs,
            "z_wheres": z_wheres,
            "gt_bbs_and_labels": gt_bbs_and_labels,
            "gt_labels_for_pred_boxes": gt_labels_for_pred_boxes,
            "pred_boxes": pred_boxes,
            "z_whats_pres_s": z_whats_pres_s
        }
        if self.return_keys is not None:
            return {k: data[k] for k in self.return_keys}
        else:
            return data

    @property
    def bb_path(self):
        return osp.join(self.bb_base_path, self.dataset_mode)
    
    @property
    def image_path(self):
        return osp.join(self.image_base_path, self.dataset_mode)
    
    @property
    def latents_path(self):
        return osp.join(self.latents_base_path, self.dataset_mode)

    def __len__(self):
        return self.image_fn_count // self.T

    def read_img(self, stack_idx, i, base_path):
        path = os.path.join(base_path, f'{stack_idx:05}_{i}.png')
        return np.array(Image.open(path).convert('RGB'))

    def read_tensor(self, stack_idx, i, base_path, postfix=None):
        path = os.path.join(base_path,
                            f'{stack_idx:05}_{i}_{postfix}.pt'
                            if postfix else f'{stack_idx:05}_{i}.pt')
        #path = os.path.join(base_path,
        #                    f'{prefix}_{stack_idx:05}_{i}.pt'
        #                    if prefix else f'{stack_idx:05}_{i}.pt')
        return torch.load(path)
    
    def read_csv(self, stack_idx, i, base_path):
        path = os.path.join(base_path, f"{stack_idx:05}_{i}.csv")
        df = pd.read_csv(path, header=None)
        df[4] = df[4].apply(lambda x: 1 if x.lower() == "m" else 0)
        label_list = label_list_for(self.game)
        df[5] = df[5].apply(lambda x: label_list.index(x))
        # convert to tensor
        return torch.tensor(df.values)

    

