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

VISUALIZE_CLASSIFIER = True #TODO: specify somewhere else

def train_classifier(cfg):
    # TODO: specify somewhere else
    data_subset_mode = "relevant"
    dataset_mode = "test"
    
    # collect the data
    relevant_labels, test_x, test_y, train_x, train_y = ZWhatDataCollector(cfg).collect_z_what_data_for_data_subset_mode(cfg, dataset_mode, data_subset_mode)
    # put all data onto cpu
    test_x, test_y, train_x, train_y = test_x.cpu(), test_y.cpu(), train_x.cpu(), train_y.cpu()

    ### K-MEANS ###
    kmeans, kmeans_clusters, kmeans_centers = create_k_means(cfg, train_x, data_subset_mode)
    # find out mapping of enumerated labels to actual labels (i.e. index in oc_atari labels)
    nn_neighbors_clf, centroids, centroid_labels = ZWhatClassifierCreator(cfg).nn_clf_based_on_k_means_centroids(kmeans, train_x, train_y, relevant_labels)
    #save centroid_labels as a csv file
    df = pd.DataFrame(centroid_labels)
    df.to_csv(f"{cfg.resume_ckpt.rsplit('/', 1)[0]}/z_what-classifier_{data_subset_mode}_centroid_labels.csv", header=False, index=True)

    ### X-MEANS ###
    #xmeans_instance, xmeans_clusters, xmeans_centers = create_x_means(cfg, train_x, data_subset_mode)

    # select which classifier to use for visualization
    clf, clusters, centers = kmeans, kmeans_clusters, kmeans_centers

    # Visualize clustering results
    if VISUALIZE_CLASSIFIER:
        visualize_classifier(cfg, dataset_mode, data_subset_mode, train_x, clusters, centers, clf)

def create_x_means(cfg, train_x, data_subset_mode):
    train_x = np.array(train_x)
    xmeans_instance = ZWhatClassifierCreator(cfg).create_x_means(train_x, kmax=3)
    ZWhatClassifierCreator(cfg).save_classifier(xmeans_instance, cfg.resume_ckpt.split("/")[-1][:-4], clf_name=f"{data_subset_mode}", folder=cfg.resume_ckpt.rsplit('/', 1)[0]) #TODO: improve
    # Extract clustering results: clusters and their centers
    clusters = xmeans_instance.get_clusters()
    centers = xmeans_instance.get_centers()
    return xmeans_instance, clusters, centers

def create_k_means(cfg, train_x, data_subset_mode):
    # create a kmeans classifier
    k_means = ZWhatClassifierCreator(cfg).create_k_means(train_x, get_moving_indices(cfg.exp_name))
    ZWhatClassifierCreator(cfg).save_classifier(k_means, cfg.resume_ckpt.split("/")[-1][:-4], clf_name=f"{data_subset_mode}", folder=cfg.resume_ckpt.rsplit('/', 1)[0]) #TODO: improve
    # collect the clusters and their centers for visualizations
    clusters = []
    for label in range(len(k_means.cluster_centers_)):
        clusters.append(np.where(k_means.labels_ == label)[0])
    centers = k_means.cluster_centers_
    return k_means, clusters, centers












    
