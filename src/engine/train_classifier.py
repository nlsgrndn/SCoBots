from model.z_what_classifier.z_what_classification import ZWhatClassifierCreator
import numpy as np
from collections import Counter
from eval.space_eval import SpaceEval
from dataset import get_dataset
from model import get_model
from eval.space_eval import SpaceEval
from dataset import get_dataset, get_dataloader
from checkpointer import Checkpointer
import os
import os.path as osp
from torch import nn
from eval.clustering_eval import ClusteringEval
from eval.classify_z_what import ZWhatEvaluator
import torch
from eval.utils import flatten
from torch.utils.tensorboard import SummaryWriter


def train_classifier(cfg):
    # model loading
    model = get_model(cfg)
    model = model.to(cfg.device)
    checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt)
    checkpointer.load_last(cfg.resume_ckpt, model, None, None, cfg.device)
    log_path = os.path.join(cfg.logdir, cfg.exp_name)
    global_step = 100000
    tb_writer = SummaryWriter(log_dir=log_path, flush_secs=30,
                           purge_step=global_step) # tb refers to tensorboard

    # SpaceEval
    evaluator = SpaceEval(cfg, tb_writer)
    dataset = get_dataset(cfg, 'test') #TODO check which dataset should be used here
    _, logs = evaluator.apply_model(dataset, cfg.device, model, global_step = None, use_global_step=False) # Data preparation

    ## ClusteringEval
    #clustering_eval = ClusteringEval("dummy")
    #z_encs, z_whats, all_labels, all_labels_moving, image_refs = clustering_eval.collect_data(logs, dataset, global_step = None, cfg = cfg) # Data preparation
    #z_whats = torch.stack(z_whats).detach().cpu() # Data preparation
    #all_labels_relevant_idx, all_labels_relevant = dataset.to_relevant(all_labels_moving) # Data preparation
    #z_whats_relevant = z_whats[flatten(all_labels_relevant_idx)] # Data preparation
#
    ## ZWhatEvaluator
    #z_what_evaluator = ZWhatEvaluator(cfg)
    #relevant_labels, test_x, test_y, train_x, train_y = z_what_evaluator.prepare_data(z_whats_relevant, flatten(all_labels_relevant)) # Data preparation

    # ClusteringEval
    clustering_eval = ClusteringEval(cfg, "dummy")
    data, _, _= clustering_eval.collect_data(logs, dataset, global_step = None) # Data preparation
    relevant_labels, test_x, test_y, train_x, train_y = data["relevant"]

    # ZWhatClassifierCreator
    clf_creator = ZWhatClassifierCreator(cfg)
    ridge_classifiers = clf_creator.create_ridge_classifiers(relevant_labels, train_x, train_y)
    k_means = clf_creator.create_k_means(train_x, relevant_labels)
    nn_clf, _, _ = clf_creator.nn_clf_based_on_k_means_centroids(k_means, train_x, train_y, relevant_labels)
    return ridge_classifiers, k_means, nn_clf,
