from model.z_what_classifier.z_what_classification import ZWhatClassifierCreator
import numpy as np
from collections import Counter
from model import get_model
from utils.checkpointer import Checkpointer
import os
import os.path as osp
from torch import nn
import torch
from eval.utils import flatten
from sklearn.decomposition import PCA
from utils.bbox_matching import match_bounding_boxes_z_what
from dataset.atari_labels import label_list_for, get_moving_indices
import pandas as pd
import random
from eval.create_z_what_dataset import ZWhatDataCollector
from classifier_visualization.classifier_vis import visualize_classifier
from eval.z_what_eval import ZWhatPlotter, ZWhatClassifierEvaluator
from PIL import Image
import joblib
from motrackers import load_classifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import create_latent_dataset
from create_latent_dataset import create_latent_dataset

def eval_classifier(cfg):

    latent_dataset = None
    classifier_path = f"{cfg.resume_ckpt.rsplit('/', 1)[0]}/z_what-classifier_relevant.joblib.pkl"
    centroids_path = f"{cfg.resume_ckpt.rsplit('/', 1)[0]}/z_what-classifier_relevant_centroid_labels.csv"
    
    clf, centroid_labels = load_classifier("dummy", centroids_path, classifier_path)
    
    # TODO: specify somewhere else
    data_subset_mode = "relevant"
    dataset_mode = "test"
    create_latent_dataset(cfg, dataset_mode)
    
    # collect the data
    x, y = ZWhatDataCollector(cfg).collect_z_what_data_for_data_subset_mode(cfg, dataset_mode, data_subset_mode)
    x, y = x.cpu(), y.cpu() # put all data onto cpu

    pred_y = clf.predict(x)
    pred_y = [centroid_labels[pred_y_i] for pred_y_i in pred_y]
 
    metric_dict = classification_report(y, pred_y, output_dict=True)
    for k, v in metric_dict.items():
        print(f"{k}: {v}")
    
    

