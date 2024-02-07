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
from dataset.atari_labels import label_list_for, get_moving_indices
import pandas as pd
import random
from dataset.atari_data_collector import AtariDataCollector
from classifier_visualization.classifier_vis import visualize_classifier, visualize_false_predictions
from PIL import Image
from create_latent_dataset import create_latent_dataset
from classifier_visualization.classifier_vis import create_cluster_folders, create_one_grid_image_for_each_cluster

def train_classifier(cfg):
    data_subset_mode = cfg.classifier.data_subset_mode
    dataset_mode = cfg.classifier.train_folder
    only_collect_first_image_of_consecutive_frames = cfg.classifier.one_image_per_sequence

    # collect the data
    create_latent_dataset(cfg, dataset_mode=dataset_mode)
    z_whats, labels = AtariDataCollector.collect_z_what_data_reshaped(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames)
    z_whats, labels = z_whats.cpu(), labels.cpu()

    if data_subset_mode == "relevant":
        relevant_labels = get_moving_indices(cfg.gamelist[0])
    elif data_subset_mode == "all":
        relevant_labels = np.arange(1, len(label_list_for(cfg.gamelist[0])))
    else:
        raise ValueError(f"Invalid data_subset_mode {data_subset_mode}")

    print("Training classifier on", len(z_whats), "z_what encodings")

    ### K-MEANS and NN CLASSIFIER based on K-MEANS ###
    kmeans, kmeans_clusters, kmeans_centers = create_k_means(cfg, z_whats, data_subset_mode)
    # find out mapping of enumerated labels to actual labels (i.e. index in oc_atari labels)
    nn_neighbors_clf, centroids, centroid_labels = create_nn_classifier(cfg, kmeans, z_whats, labels, relevant_labels, data_subset_mode)
    #save centroid_labels as a csv file
    df = pd.DataFrame(centroid_labels)
    df.to_csv(f"{cfg.resume_ckpt.rsplit('/', 1)[0]}/z_what-classifier_{data_subset_mode}_kmeans_centroid_labels.csv", header=False, index=True)

    ### X-MEANS ###
    xmeans_instance, xmeans_clusters, xmeans_centers = create_x_means(cfg, z_whats, data_subset_mode)

    if cfg.classifier.visualize:
        # select which classifier to use for visualization
        clf, clusters, centers = kmeans, kmeans_clusters, kmeans_centers
        visualize_classifier(cfg, dataset_mode, data_subset_mode, z_whats, labels, clusters, centers, clf, centroid_labels, relevant_labels, only_collect_first_image_of_consecutive_frames)


def create_x_means(cfg, train_x, data_subset_mode):
    train_x = np.array(train_x)
    xmeans_instance = ZWhatClassifierCreator(cfg).create_x_means(train_x, kmax=3)
    save_classifier(cfg, xmeans_instance, "xmeans", data_subset_mode)
    # Extract clustering results: clusters and their centers
    clusters = xmeans_instance.get_clusters()
    centers = xmeans_instance.get_centers()
    return xmeans_instance, clusters, centers

def create_k_means(cfg, train_x, data_subset_mode):

    # create a kmeans classifier
    k_means = ZWhatClassifierCreator(cfg).create_k_means(train_x, get_moving_indices(cfg.exp_name))
    save_classifier(cfg, k_means, "kmeans", data_subset_mode)
    # collect the clusters and their centers for visualizations
    clusters = []
    for label in range(len(k_means.cluster_centers_)):
        clusters.append(np.where(k_means.labels_ == label)[0])
    centers = k_means.cluster_centers_
    return k_means, clusters, centers

def create_nn_classifier(cfg, kmeans, train_x, train_y, relevant_labels, data_subset_mode):
    # create a nearest neighbor classifier
    nn_neighbors_clf, centroids, centroid_labels = ZWhatClassifierCreator(cfg).nn_clf_based_on_k_means_centroids(kmeans, train_x, train_y, relevant_labels)
    save_classifier(cfg, nn_neighbors_clf, "nn", data_subset_mode)
    return nn_neighbors_clf, centroids, centroid_labels

def save_classifier(cfg, clf, clf_name, data_subset_mode):
    ZWhatClassifierCreator(cfg).save_classifier(
        clf=clf,
        clf_name=f"{clf_name}",
        folder=cfg.resume_ckpt.rsplit('/', 1)[0],
        data_subset_mode=data_subset_mode
    )
    












    
