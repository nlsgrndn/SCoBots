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
from torch.utils.tensorboard import SummaryWriter
from utils.bbox_matching import match_bounding_boxes_z_what
from dataset.atari_labels import label_list_for, get_moving_indices
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.xmeans import xmeans, splitting_type
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import SIMPLE_SAMPLES
import pandas as pd
import random
from PIL import Image
from dataset.z_what import Atari_Z_What
from torch.utils.data import DataLoader



def train_classifier(cfg):
    # model loading
    model = collect_model(cfg)
    data_category = "relevant"
    data_source = "dataloader"
    relevant_labels, test_x, test_y, train_x, train_y = collect_dataset(cfg, model, data_category=data_category, data_source=data_source)

    k_means = ZWhatClassifierCreator(cfg).create_k_means(train_x, get_moving_indices(cfg.exp_name))
    k_means_labels = k_means.labels_
    clusters = []
    for label in range(len(k_means.cluster_centers_)):
        clusters.append(np.where(k_means_labels == label)[0])
    centers = k_means.cluster_centers_
    
    nn_neighbors_clf, centroids, centroid_labels = ZWhatClassifierCreator(cfg).nn_clf_based_on_k_means_centroids(k_means, train_x, train_y, relevant_labels)
    filter_str = "filtered" if data_category == "relevant" else "unfiltered"
    ZWhatClassifierCreator(cfg).save_classifier(k_means, cfg.resume_ckpt.split("/")[-1][:-4], clf_name=filter_str)
    print("centroid_labels", centroid_labels)


    
    #train_x = np.array(train_x)
    #image_refs = image_refs[:len(train_x)]
    #xmeans_instance = ZWhatClassifierCreator(cfg).create_x_means(train_x, kmax=3)
    #filter_str = "filtered" if data_category == "relevant" else "unfiltered"
    #ZWhatClassifierCreator(cfg).save_classifier(xmeans_instance, cfg.resume_ckpt.split("/")[-1][:-4], clf_name=f"xmeans_{filter_str}")  
#
    ## Extract clustering results: clusters and their centers
    #clusters = xmeans_instance.get_clusters()
    #centers = xmeans_instance.get_centers()
    # Visualize clustering results
    visualizer = cluster_visualizer()
    sample, centers, dim_name = perform_dimensionality_reduction(train_x, centers)
    visualizer.append_clusters(clusters, sample)
    visualizer.append_cluster(centers, None, marker='*', markersize=10)
    visualizer.save('clf.png')

    clf = k_means # x_means_instance
    y_pred_train = np.array(clf.predict(train_x))
    pred_vs_gt = np.stack((y_pred_train, train_y), axis=1)
    #df = pd.DataFrame( dict(image_refs=image_refs_train, y_pred=y_pred_train, y_gt=train_y))

    #cluster_folders = visualize_clusters(cfg, df, pred_vs_gt)
    #create_one_grid_image_for_each_cluster(cluster_folders)

def collect_model(cfg):
    model = get_model(cfg)
    model = model.to(cfg.device)
    checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt)
    checkpointer.load_last(cfg.resume_ckpt, model, None, None, cfg.device)
    return model

def collect_dataset(cfg, model, data_category, data_source):
    #dataset = get_dataset(cfg, 'test') #TODO check which dataset should be used here
    if data_source == "dataloader":
        relevant_labels, test_x, test_y, train_x, train_y= collect_data_using_dataloader(cfg, data_category=data_category)
    #elif data_source == "files":
    #    relevant_labels, test_x, test_y, train_x, train_y, image_refs_train, image_refs_test = collect_data_using_files(cfg, data_category=data_category)
    #elif data_source == "eval_classes":
    #    relevant_labels, test_x, test_y, train_x, train_y = collect_data_using_eval_classes(cfg, model, dataset, global_step = None, data_category=data_category)
    else:
        raise ValueError(f"Unknown data source {data_source}")

    return relevant_labels, test_x, test_y, train_x, train_y #, image_refs_train, image_refs_test


def collect_data_using_dataloader(cfg, data_category):
    boxes_subset = data_category
    dataset_mode = "test" # TODO configure somewhere else
    atari_z_what_dataset = Atari_Z_What(cfg, dataset_mode, boxes_subset, return_keys = ["z_whats_pres_s", "gt_labels_for_pred_boxes", "imgs"])
    atari_z_what_dataloader = DataLoader(atari_z_what_dataset, batch_size=1, shuffle=False, num_workers=0) #batch_size must be 1
    z_whats = []
    labels = []
    #images = []
    for batch in atari_z_what_dataloader:
        curr_z_whats = batch["z_whats_pres_s"]
        curr_labels = batch["gt_labels_for_pred_boxes"]
        z_whats.extend([curr_z_whats[i][0] for i in range(len(curr_z_whats))]) #[0] because batch_size = 1 and dataloader implicitly adds a batch dimension
        labels.extend([curr_labels[i][0] for i in range(len(curr_labels))]) #[0] because batch_size = 1 and dataloader implicitly adds a batch dimension

        #curr_images = batch["imgs"][0] #[0] because batch_size = 1 and dataloader implicitly adds a batch dimension
        #images.extend(curr_images)
    z_whats = torch.cat(z_whats, dim=0)
    labels= torch.cat(labels, dim=0)
    labels = labels.squeeze(-1)
    relevant_labels, test_x, test_y, train_x, train_y = prepare_data(z_whats, labels)
    #images = torch.stack(images, dim=0)
    return relevant_labels, test_x, test_y, train_x, train_y #, images[:len(train_x)], images[len(train_x):]

def prepare_data(z_what, labels,):
    c = Counter(labels.tolist() if labels is not None else [])
    # Initialization stuff
    relevant_labels = list(c.keys()) #TODO: remove or allow to specify subset of labels
    # Filter out the irrelevant labels
    z_what, labels = only_keep_relevant_data(z_what, labels, relevant_labels)
    # Split the data into train and test
    train_x, train_y, test_x, test_y = train_test_split(z_what, labels, train_portion=0.9)
    if len(c) < 2 or len(torch.unique(train_y)) < 2:
        return None, None, None, None, None
    
    return relevant_labels, test_x, test_y, train_x, train_y
    
def train_test_split(z_what, labels, train_portion=0.9):
    nb_sample = int(train_portion * len(labels))
    train_x = z_what[:nb_sample]
    train_y = labels[:nb_sample]
    test_x = z_what[nb_sample:]
    test_y = labels[nb_sample:]
    return train_x, train_y, test_x, test_y

def only_keep_relevant_data(z_what, labels, relevant_labels):
    relevant = torch.zeros(labels.shape, dtype=torch.bool)
    for rl in relevant_labels:
        relevant |= labels == rl
    return z_what[relevant], labels[relevant]


def visualize_clusters(cfg, df, pred_vs_gt):
    # save all images of all present combinations of (y_pred and y_gt) in a separate folder
    #for y_pred, y_gt in Counter([tuple(x) for x in pred_vs_gt]):
    #    df_filtered = df[(df.y_pred == y_pred) & (df.y_gt == y_gt)]
    #    for image_ref in df_filtered.image_refs:
    #        image_path = f"labeled/{cfg.exp_name}/objects/{image_ref}"
    #        image = Image.open(image_path)
    #        label = label_list_for(cfg.exp_name)[int(y_gt)]
    #        folder = f"labeled/{cfg.exp_name}/objects/cluster{int(y_pred)}_gt{label}"
    #        if not os.path.exists(folder):
    #            os.makedirs(folder)
    #        image.save(folder + f"/{image_ref}")

    # remove all existing cluster folders
    for folder in os.listdir(f"labeled/{cfg.exp_name}/objects"):
        if "cluster" in folder:
            os.system(f"rm -rf labeled/{cfg.exp_name}/objects/{folder}")

    folders = []
    for y_pred, y_gt in Counter([tuple(x) for x in pred_vs_gt]):
        df_filtered = df[(df.y_pred == y_pred) & (df.y_gt == y_gt)]
        for image_ref in df_filtered.image_refs:
            image_path = f"labeled/{cfg.exp_name}/objects/{image_ref}"
            image = Image.open(image_path)
            label = label_list_for(cfg.exp_name)[int(y_gt)]
            folder = f"labeled/{cfg.exp_name}/objects/cluster{int(y_pred)}"
            folders.append(folder)
            if not os.path.exists(folder):
                os.makedirs(folder)
            image.save(folder + f"/{image_ref}")
    return folders

import math
def create_one_grid_image_for_each_cluster(cluster_folders):

    # open each folder iteratively
    for folder in cluster_folders:
        x_size, y_size = (20,20)
        # create grid image containing each image in the folder
        image_files = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]

        if not image_files:
            print(f"No images found in {folder}")
            continue

        # Create a new blank image to accommodate all the small images
        grid_size = (int(math.sqrt(len(image_files))), int(math.ceil(len(image_files) / math.sqrt(len(image_files)))))
        grid_image = Image.new('RGB', (grid_size[0] * x_size, grid_size[1] * y_size))

        # Iterate through each image and paste it into the grid
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(folder, image_file)
            img = Image.open(image_path)

            # Resize the image to fit in the grid (adjust as needed)
            img = img.resize((x_size, y_size), Image.ANTIALIAS)

            # Calculate the position to paste the image
            row = i // grid_size[0]
            col = i % grid_size[0]

            # Paste the resized image into the grid
            grid_image.paste(img, (col * x_size,  row * y_size))

        # Save the grid image for the current cluster
        base_folder, cluster = folder.rsplit("/", 1)
        base_folder = base_folder + "/cluster_grid_images"
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        grid_image.save(os.path.join(base_folder,  f"{cluster}_grid.png"))


def perform_dimensionality_reduction(z_what, centroids,):
    # perform PCA or TSNE
    pca = PCA(n_components=2)
    z_what_emb = pca.fit_transform(z_what)
    centroid_emb = pca.transform(np.array(centroids))
    dim_name = "PCA"
    return z_what_emb, centroid_emb, dim_name



    
