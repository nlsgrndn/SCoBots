from .atari import Atari
from .obj3d import Obj3D
from torch.utils.data import DataLoader
from .atari_labels import label_list_for
import torch.utils.data as data_utils
import torch


__all__ = ['get_dataset', 'get_dataloader', 'get_label_list']


def get_dataset(cfg, dataset_mode):
    assert dataset_mode in ['train', 'val', 'test']
    if cfg.dataset == 'ATARI':
        return Atari(cfg, dataset_mode)
    elif cfg.dataset == 'OBJ3D_SMALL':
        return Obj3D(cfg.dataset_roots.OBJ3D_SMALL, dataset_mode)
    elif cfg.dataset == 'OBJ3D_LARGE':
        return Obj3D(cfg.dataset_roots.OBJ3D_LARGE, dataset_mode)


def get_dataloader(cfg, dataset_mode, no_shuffle_overwrite = False):
    assert dataset_mode in ['train', 'val', 'test']
    if dataset_mode == 'train': #TODO improve this part of implementation (also look at eval_cfg)
        batch_size = getattr(cfg, dataset_mode).batch_size
        num_workers = getattr(cfg, dataset_mode).num_workers
    elif dataset_mode == 'val':
        batch_size = cfg.eval_cfg.val.batch_size
        num_workers = cfg.eval_cfg.val.num_workers
    elif dataset_mode == 'test':
        batch_size = cfg.eval_cfg.test.batch_size 
        num_workers = cfg.eval_cfg.test.num_workers
    shuffle = True if dataset_mode == 'train' else False
    if no_shuffle_overwrite:
        shuffle = False

    dataset = get_dataset(cfg, dataset_mode)
    #dataset = data_utils.Subset(dataset, torch.arange(dataset_size)) #TODO: check why this line was here
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


    return dataloader


def get_label_list(cfg):
    game = cfg.gamelist[0]
    return label_list_for(game)
