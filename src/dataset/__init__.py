from .atari import Atari
from .obj3d import Obj3D
from torch.utils.data import DataLoader
from .labels import label_list_pacman, label_list_carnival, label_list_pong, label_list_space_invaders,\
    label_list_tennis, label_list_boxing, label_list_riverraid, label_list_air_raid
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


def get_dataloader(cfg, dataset_mode):
    assert dataset_mode in ['train', 'val', 'test']
    batch_size = getattr(cfg, dataset_mode).batch_size
    shuffle = True if dataset_mode == 'train' else False
    num_workers = getattr(cfg, dataset_mode).num_workers
    dataset = get_dataset(cfg, dataset_mode)
    #dataset = data_utils.Subset(dataset, torch.arange(dataset_size)) #TODO: check why this line was here
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


    return dataloader


def get_label_list(cfg):
    game = cfg.gamelist[0]
    if "MsPacman" in game:
        return label_list_pacman
    elif "Carnival" in game:
        return label_list_carnival
    elif "SpaceInvaders" in game:
        return label_list_space_invaders
    elif "Pong" in game:
        return label_list_pong
    elif "Boxing" in game:
        return label_list_boxing
    elif "Tennis" in game:
        return label_list_tennis
    elif "Riverraid" in game:
        return label_list_riverraid
    elif "Airraid" in game:
        return label_list_air_raid
    else:
        raise ValueError(f"get_label_list failed for game {game}")
