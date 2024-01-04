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

# TODO: currently unused, Atari_Z_What is used instead, maybe reactivate and let Atari_Z_What inherit from this class 
class Atari(Dataset):
    def __init__(self, cfg, dataset_mode, nr_consecutive_frames=4):
        self.game = cfg.gamelist[0]
        
        assert dataset_mode in ['train', 'val', 'test'], f'Invalid dataset mode "{dataset_mode}"'
        dataset_mode = 'validation' if dataset_mode == 'val' else dataset_mode
        self.dataset_mode = dataset_mode

        self.T = nr_consecutive_frames
        
        #folder paths
        img_folder = "space_like"
        self.image_base_path = osp.join(cfg.dataset_roots.ATARI, cfg.gamelist[0], img_folder)
        self.motion_base_path = osp.join(cfg.dataset_roots.ATARI, cfg.gamelist[0], cfg.arch.motion_kind)
        self.bb_base_path = osp.join(cfg.dataset_roots.ATARI, cfg.gamelist[0], 'bb')

        self.image_fn_count = len([True for img in os.listdir(self.image_path) if img.endswith(".png")])
        
        self.arch = cfg.arch

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
        base_path = self.image_path
        imgs = torch.stack([transforms.ToTensor()(self.read_img(stack_idx, i)) for i in range(self.T)])
        
        base_path = self.motion_path
        motion = torch.stack([self.read_tensor(stack_idx, i, base_path, postfix=f'{self.arch.img_shape[0]}') for i in range(self.T)])
        motion = (motion > motion.mean() * 0.1).float()
        motion_z_pres = torch.stack([self.read_tensor(stack_idx, i, base_path, postfix="z_pres") for i in range(self.T)])
        motion_z_where = torch.stack([self.read_tensor(stack_idx, i, base_path, postfix="z_where") for i in range(self.T)])
        return imgs, (motion > motion.mean() * 0.1).float(), motion_z_pres, motion_z_where

    def __len__(self):
        return self.image_fn_count // self.T

    def read_img(self, stack_idx, i, base_path):
        path = os.path.join(base_path, f'{stack_idx:05}_{i}.png')
        return np.array(Image.open(path).convert('RGB'))

    def read_tensor(self, stack_idx, i, base_path, postfix=None):
        path = os.path.join(base_path,
                            f'{stack_idx:05}_{i}_{postfix}.pt'
                            if postfix else f'{stack_idx:05}_{i}.pt')
        return torch.load(path)

    

