import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
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
from space_and_moc_utils.bbox_matching import get_label_of_best_matching_gt_bbox
from dataset.atari_labels import label_list_for, no_label_str, filter_relevant_boxes_masks, get_moving_indices
from  space_models.space.postprocess_latent_variables import convert_to_boxes, retrieve_latent_repr_from_logs
    
class Atari_Z_What(Dataset):
    def __init__(self, cfg, dataset_mode, boxes_subset="all", return_keys=None, nr_consecutive_frames=4):
        self.game = cfg.gamelist[0]

        assert dataset_mode in ['train', 'val', 'test'], f'Invalid dataset mode "{dataset_mode}"'
        dataset_mode = 'validation' if dataset_mode == 'val' else dataset_mode
        self.dataset_mode = dataset_mode

        self.T = nr_consecutive_frames

        # folder paths
        img_folder = "space_like"
        self.image_base_path = osp.join(cfg.dataset_roots.ATARI, self.game, img_folder)
        self.bb_base_path = osp.join(cfg.dataset_roots.ATARI, self.game, "bb")
        self.motion_base_path = osp.join(cfg.dataset_roots.ATARI, cfg.gamelist[0], cfg.arch.motion_kind)
        self.latents_base_path = osp.join(cfg.dataset_roots.ATARI, self.game, "latents")

        self.image_fn_count = len([True for img in os.listdir(self.image_path) if img.endswith(".png")])

        max_num_of_different_samples_for_dataset_mode = {
            "train": cfg.dataset_size_cfg.max_num_of_different_samples_for_dataset_mode.train,
            "validation": cfg.dataset_size_cfg.max_num_of_different_samples_for_dataset_mode.val, 
            "test": cfg.dataset_size_cfg.max_num_of_different_samples_for_dataset_mode.test
        }
                                                         
        self.max_num_samples = max_num_of_different_samples_for_dataset_mode[dataset_mode]
        self.len = min(self.image_fn_count // self.T, self.max_num_samples)
        if self.len < self.max_num_samples:
            print(f"Warning: available number of samples for {dataset_mode} is {self.len} but max_num_samples is {self.max_num_samples}.")

        self.boxes_subset = boxes_subset # "all", "relevant"
        self.return_keys = return_keys

        self.arch = cfg.arch

    @property
    def bb_path(self):
        return osp.join(self.bb_base_path, self.dataset_mode)
    
    @property
    def image_path(self):
        return osp.join(self.image_base_path, self.dataset_mode)
    
    @property
    def latents_path(self):
        return osp.join(self.latents_base_path, self.dataset_mode)
    
    @property
    def motion_path(self):
        return osp.join(self.motion_base_path, self.dataset_mode)

    def __getitem__(self, stack_idx): # (T, ...) where T is number of consecutive frames and ... represents the actual dimensions of the data
        
        base_path = self.image_path
        imgs = torch.stack([transforms.ToTensor()(self.read_img(stack_idx, i, base_path)) for i in range(self.T)])
        
        if self.return_keys == ["imgs"]:
            return {"imgs": imgs}

        base_path = self.motion_path
        motion = torch.stack([self.read_tensor(stack_idx, i, base_path, postfix=f'{self.arch.img_shape[0]}') for i in range(self.T)])
        motion = (motion > motion.mean() * 0.1).float() # Why???
        motion_z_pres = torch.stack([self.read_tensor(stack_idx, i, base_path, postfix="z_pres") for i in range(self.T)])
        motion_z_where = torch.stack([self.read_tensor(stack_idx, i, base_path, postfix="z_where") for i in range(self.T)])
        
        # TODO improve this part of implementation
        # early return if only imgs, motion, motion_z_pres, motion_z_where are requested
        if set(self.return_keys) == set(["imgs", "motion", "motion_z_pres", "motion_z_where"]):
            return {
                "imgs": imgs,
                "motion": motion,
                "motion_z_pres": motion_z_pres,
                "motion_z_where": motion_z_where
            }

        base_path = self.latents_path
        z_whats = self.read_tensor_of_complete_T_dim(stack_idx, base_path, "z_what")
        z_pres_probs = self.read_tensor_of_complete_T_dim(stack_idx, base_path, "z_pres_prob")
        z_wheres = self.read_tensor_of_complete_T_dim(stack_idx, base_path, "z_where")
        z_pres_s = z_pres_probs > 0.5

        base_path = self.bb_path
        gt_bbs_and_labels = [self.read_csv(stack_idx, i, base_path) for i in range(self.T)] # can't be stacked because of varying number of objects
        
        ## boxes_subset == "moving" is deprecated
        #if self.boxes_subset == "moving": # modify labels of static objects to "no_label"
        #    for i in range(self.T):
        #        mask = gt_bbs_and_labels[i][:, 4] == 0 # static objects
        #        index_for_no_label = label_list_for(self.game).index(no_label_str)
        #        gt_bbs_and_labels[i][mask, 5] = index_for_no_label

        if self.boxes_subset == "relevant":  # remove non-moving gt boxes
            gt_bbs_and_labels = [gt_bbs_and_labels[i][gt_bbs_and_labels[i][:, 4] == 1] for i in range(self.T)]

        pred_boxes = convert_to_boxes(z_wheres, z_pres_s, z_pres_probs, with_conf=True) # list of arrays of shape (N, 4) where N is number of objects in that frame

        z_whats_pres_s = []
        for i in range(self.T):
            z_whats_pres_s.append(z_whats[i][z_pres_s[i]])
        
        gt_labels_for_pred_boxes = []
        for i in range(self.T):
            if isinstance(gt_bbs_and_labels[i], torch.Tensor) and gt_bbs_and_labels[i].dtype == torch.float64:
                gt_bbs_and_labels[i] = gt_bbs_and_labels[i].to(torch.float)
            gt_labels_for_pred_boxes.append(get_label_of_best_matching_gt_bbox(torch.Tensor(gt_bbs_and_labels[i]), torch.Tensor(pred_boxes[i])).reshape(-1, 1).to(torch.int)) 

        if self.boxes_subset == "relevant":
            masks = filter_relevant_boxes_masks(self.game, pred_boxes, None)
            for i in range(self.T):
                pred_boxes[i] = pred_boxes[i][masks[i]]
                gt_labels_for_pred_boxes[i] = gt_labels_for_pred_boxes[i][masks[i]]
                z_whats_pres_s[i] = z_whats_pres_s[i][masks[i]]

        data = {
            "imgs": imgs,
            "motion": motion,
            "motion_z_pres": motion_z_pres,
            "motion_z_where": motion_z_where,
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

    def __len__(self):
        return self.len

    def read_img(self, stack_idx, i, base_path):
        path = os.path.join(base_path, f'{stack_idx:05}_{i}.png')
        return np.array(Image.open(path).convert('RGB'))

    def read_tensor(self, stack_idx, i, base_path, postfix=None):
        path = os.path.join(base_path,
                            f'{stack_idx:05}_{i}_{postfix}.pt'
                            if postfix else f'{stack_idx:05}_{i}.pt')
        return torch.load(path)
    
    def read_tensor_of_complete_T_dim(self, stack_idx, base_path, postfix=None):
        infix = f"0to{self.T-1}" if self.T > 1 else "0"
        path = os.path.join(base_path,
                            f'{stack_idx:05}_{infix}_{postfix}.pt'
                            if postfix else f'{stack_idx:05}_{infix}.pt')
        return torch.load(path)
    
    def read_csv(self, stack_idx, i, base_path):
        path = os.path.join(base_path, f"{stack_idx:05}_{i}.csv")
        df = pd.read_csv(path, header=None)
        df[4] = df[4].apply(lambda x: 1 if x.lower() == "m" else 0)
        label_list = label_list_for(self.game)
        df[5] = df[5].apply(lambda x: label_list.index(x))
        # convert to tensor
        return torch.tensor(df.values)

    

