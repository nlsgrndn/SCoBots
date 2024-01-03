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
from utils.bbox_matching import match_bbs
from dataset.atari_labels import label_list_for, no_label_str
from utils.bbox_matching import match_bbs


class Atari(Dataset):
    def __init__(self, cfg, dataset_mode, nr_consecutive_frames=4):
        assert dataset_mode in ['train', 'val', 'test'], f'Invalid dataset mode "{dataset_mode}"'
        dataset_mode = 'validation' if dataset_mode == 'val' else dataset_mode

        img_folder = "space_like"
        self.image_base_path = osp.join(cfg.dataset_roots.ATARI, cfg.gamelist[0], img_folder)
        self.motion_base_path = osp.join(cfg.dataset_roots.ATARI, cfg.gamelist[0], cfg.arch.motion_kind)
        self.bb_base_path = osp.join(cfg.dataset_roots.ATARI, cfg.gamelist[0], 'bb')

        self.dataset_mode = dataset_mode
        self.game = cfg.gamelist[0]
        self.arch = cfg.arch
        self.motion = cfg.arch.motion
        self.motion_kind = cfg.arch.motion_kind
        self.image_fn_count = len([True for img in os.listdir(self.image_path) if img.endswith(".png")])
        self.T = nr_consecutive_frames

    @property
    def bb_path(self):
        return osp.join(self.bb_base_path, self.dataset_mode)
    
    @property
    def motion_path(self):
        return osp.join(self.motion_base_path, self.dataset_mode)
    
    @property
    def image_path(self):
        return osp.join(self.image_base_path, self.dataset_mode)

    def __getitem__(self, stack_idx): # (T, ...) where T is number of consecutive frames and ... represents the actual dimensions of the data
        imgs = torch.stack([transforms.ToTensor()(self.read_img(stack_idx, i)) for i in range(self.T)])
        motion = torch.stack([self.read_tensor(stack_idx, i, postfix=f'{self.arch.img_shape[0]}') for i in range(self.T)])
        motion_z_pres = torch.stack([self.read_tensor(stack_idx, i, postfix="z_pres") for i in range(self.T)])
        motion_z_where = torch.stack([self.read_tensor(stack_idx, i, postfix="z_where") for i in range(self.T)])
        return imgs, (motion > motion.mean() * 0.1).float(), motion_z_pres, motion_z_where

    def __len__(self):
        return self.image_fn_count // self.T

    def read_img(self, stack_idx, i):
        path = os.path.join(self.image_path, f'{stack_idx:05}_{i}.png')
        return np.array(Image.open(path).convert('RGB'))

    def read_tensor(self, stack_idx, i, postfix=None):
        path = os.path.join(self.motion_path,
                            f'{stack_idx:05}_{i}_{postfix}.pt'
                            if postfix else f'{stack_idx:05}_{i}.pt')
        return torch.load(path)



    def get_labels_impl(self, gt_bbs_batch,  boxes_batch, bbox_matching_method = match_bbs):
        idx = 0
        labels = []
        sub_labels = []
        for gt_bbs, boxes in zip(gt_bbs_batch, boxes_batch):
            sub_labels.append(bbox_matching_method(gt_bbs, self.game, boxes, no_label_str))
            idx += 1
            if idx == self.T:
                labels.append(sub_labels)
                sub_labels = []
                idx = 0
        return labels
    
    def get_gt_bbs(self, batch_start, batch_end):
        bbs = []
        for stack_idx in range(batch_start, batch_end):
            for img_idx in range(self.T):
                bbs.append(pd.read_csv(os.path.join(self.bb_path, f"{stack_idx:05}_{img_idx}.csv"), header=None))
        return bbs
    
    def get_labels(self, gt_bbs, boxes_batch):
        return self.get_labels_impl(gt_bbs,  boxes_batch, self.match_labels)

    def to_relevant(self, labels_moving):
        return to_relevant(self.game, labels_moving)

    def filter_relevant_boxes(self, boxes_batch, boxes_gt):
        return filter_relevant_boxes(self.game, boxes_batch, boxes_gt)

    

