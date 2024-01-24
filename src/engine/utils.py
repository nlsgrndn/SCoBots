import os
import argparse
from argparse import ArgumentParser

from torch import nn
import os.path as osp
from configs.config import cfg
from model import get_model
from solver import get_optimizers
from utils.checkpointer import Checkpointer

import joblib
import pandas as pd


def get_config():
    parser = ArgumentParser()
    parser.add_argument(
        '--task',
        type=str,
        default='train',
        metavar='TASK',
        help='What to do. See engine'
    )
    parser.add_argument(
        '--config-file',
        type=str,
        default='',
        metavar='FILE',
        help='Path to config file'
    )

    parser.add_argument(
        '--resume_ckpt',
        help='Provide a checkpoint to restart training from',
        default=''
    )

    parser.add_argument(
        '--arch-type',
        help='architecture type',
        choices=['baseline', '+m', '+moc'],
        default= "+moc",
    )

    parser.add_argument(
        'opts',
        help='Modify config options using the command line',
        default=None,
        nargs=argparse.REMAINDER
    )
    # example usage for opts: python main.py arch.area_object_weight 0.0 arch.motion_input False

    args = parser.parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    if cfg.model.lower() in ['lrspace', "lrtcspace", "tclrspace"]:
        cfg.arch = cfg.lr_arch
    # Use config file name as the default experiment name
    if cfg.exp_name == '':
        if args.config_file:
            cfg.exp_name = os.path.splitext(os.path.basename(args.config_file))[0]
        else:
            raise ValueError('exp_name cannot be empty without specifying a config file')
    # add desc of arch type for eval
    if args.arch_type is None or args.arch_type == 'baseline':
        cfg.arch.area_object_weight = 0.0
        cfg.arch.motion_weight = 0.0
    elif args.arch_type == "+m": #
        cfg.arch.area_object_weight = 0.0
    elif args.arch_type == "+moc": #
        cfg.arch.area_object_weight = 10.0

    if args.resume_ckpt != '':
        cfg.resume_ckpt = args.resume_ckpt
    arch_type = '' if args.arch_type == "baseline" else args.arch_type
    cfg.arch_type = args.arch_type

    #if args.resume_ckpt == '':
    #    cfg.resume_ckpt = f"../trained_models/{cfg.exp_name}_space{arch_type}_seed{cfg.seed}.pth"
    #    print(f"Using checkpoint from {cfg.resume_ckpt}")

    import torch
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    np.random.seed(cfg.seed)

    return cfg, args.task


def get_config_v2(config_path):
    cfg.merge_from_file(config_path)
    return cfg


def print_info(cfg):
    print('Experiment name:', cfg.exp_name)
    print('Dataset:', cfg.dataset)
    print('Model name:', cfg.model)
    print('Resume:', cfg.resume)
    if cfg.resume:
        print('Checkpoint:', cfg.resume_ckpt if cfg.resume_ckpt else 'last checkpoint')
    print('Using device:', cfg.device)
    if 'cuda' in cfg.device:
        print('Using parallel:', cfg.parallel)
    if cfg.parallel:
        print('Device ids:', cfg.device_ids)
    suffix = cfg.gamelist[0]
    print(f"Using Game {suffix}")


def load_model(cfg, mode):
    model = get_model(cfg)
    model = model.to(cfg.device)
    checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt,)
    checkpoint = None

    if mode == "eval":
        optimizer_fg, optimizer_bg = None, None
        if cfg.resume_ckpt: # Note empty string is False
            checkpoint = checkpointer.load(cfg.resume_ckpt, model, None, None, cfg.device)
        elif cfg.eval.checkpoint == 'last':
            checkpoint = checkpointer.load_last('', model, None, None, cfg.device)
        elif cfg.eval.checkpoint == 'best':
            checkpoint = checkpointer.load_best(cfg.eval.metric, model, None, None, cfg.device)
    elif mode == "train":
        optimizer_fg, optimizer_bg = get_optimizers(cfg, model)
        if cfg.resume: # whether to resume training from a checkpoint
            checkpoint = checkpointer.load_last(cfg.resume_ckpt, model, optimizer_fg, optimizer_bg, cfg.device) #loads cfg.resume_ckpt if not empty, else loads last checkpoint

    
    if cfg.parallel:
        model = nn.DataParallel(model, device_ids=cfg.device_ids)
    return model, optimizer_fg, optimizer_bg, checkpointer, checkpoint

def load_classifier(folder_path=None, clf_name="kmeans", data_subset_mode="relevant"):
    classifier_path = f"{folder_path}/z_what-classifier_{data_subset_mode}_{clf_name}.joblib.pkl"
    classifier = joblib.load(classifier_path)

    centroid_labels_dict = None
    if clf_name == "kmeans":
        centroid_labels_path = f"{folder_path}/z_what-classifier_{data_subset_mode}_{clf_name}_centroid_labels.csv"
        centroid_labels = pd.read_csv(centroid_labels_path, header=None, index_col=0)
        centroid_labels_dict = centroid_labels.iloc[:,0].to_dict()

    return classifier, centroid_labels_dict