from model.z_what_classifier.z_what_classification import ZWhatClassifierCreator
import numpy as np
from collections import Counter
from eval.space_eval import SpaceEval
from dataset import get_dataset
from model import get_model
from eval.space_eval import SpaceEval
from dataset import get_dataset, get_dataloader
from utils.checkpointer import Checkpointer
import os
import os.path as osp
from torch import nn
from eval.clustering_eval import ClusteringEval
from eval.classify_z_what import ZWhatEvaluator
import torch
from eval.utils import flatten
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
from utils.bbox_matching import match_bounding_boxes_z_what
from dataset.atari_labels import label_list_for
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.xmeans import xmeans, splitting_type
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import SIMPLE_SAMPLES
import pandas as pd
import random
from PIL import Image



def train_classifier(cfg):
    # model loading
    model = get_model(cfg)
    model = model.to(cfg.device)
    checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt)
    checkpointer.load_last(cfg.resume_ckpt, model, None, None, cfg.device)

    # Dataset
    dataset = get_dataset(cfg, 'val') #TODO check which dataset should be used here
    data_category = "all"
    data_source = "files" #"eval_classes" or "files"
    if data_source == "files":
        (relevant_labels, test_x, test_y, train_x, train_y) , image_refs = collect_data_using_files(cfg, data_category=data_category)
    elif data_source == "eval_classes":
        relevant_labels, test_x, test_y, train_x, train_y = collect_data_using_eval_classes(cfg, model, dataset, global_step = None, data_category=data_category)
    else:
        raise ValueError(f"Unknown data source {data_source}")
    print("dataset has length", len(dataset))
    print("train_x has length", len(train_x))
    
    # ZWhatClassifierCreator
    #clf_creator = ZWhatClassifierCreator(cfg)
    #ridge_classifiers = clf_creator.create_ridge_classifiers(relevant_labels, train_x, train_y)
    #k_means = clf_creator.create_k_means(train_x, relevant_labels)
    #nn_clf, _, _ = clf_creator.nn_clf_based_on_k_means_centroids(k_means, train_x, train_y, relevant_labels)
    # generate random number without numpy
    seed = random.randint(0, 100000)
    np.random.seed(seed)
    
    image_refs = image_refs[:len(train_x)]
    #pca = PCA(n_components=0.6, svd_solver='full')
    #train_x = pca.fit_transform(train_x)
    #print(pca.components_.shape)
    sample = np.array(train_x)
    xmeans_instance, centers, _ = ZWhatClassifierCreator(cfg).create_x_means(sample, seed=seed)
    print(train_y)
    y_pred_train = np.array(xmeans_instance.predict(train_x))
    pred_vs_gt = np.stack((y_pred_train, train_y), axis=1)
    print(Counter([tuple(x) for x in pred_vs_gt]))
    df = pd.DataFrame( dict(image_refs=image_refs, y_pred=y_pred_train, y_gt=train_y))
    print("seed", seed)
    print("number of clusters", len(centers))
    # save all images of all present combinations of (y_pred and y_gt) in a separate folder
    for y_pred, y_gt in Counter([tuple(x) for x in pred_vs_gt]):
        df_filtered = df[(df.y_pred == y_pred) & (df.y_gt == y_gt)]
        for image_ref in df_filtered.image_refs:
            image_path = f"labeled/{cfg.exp_name}/objects/{image_ref}"
            image = Image.open(image_path)
            label = label_list_for(cfg.exp_name)[int(y_gt)]
            folder = f"labeled/{cfg.exp_name}/objects/cluster{int(y_pred)}_gt{label}"
            if not os.path.exists(folder):
                os.makedirs(folder)
            image.save(folder + f"/{image_ref}")

def collect_data_using_eval_classes(cfg, model, dataset, global_step, data_category="all"):
    print("Collecting data using eval classes")
    # SpaceEval
    log_path = os.path.join(cfg.logdir, cfg.exp_name)
    global_step = 100000
    tb_writer = SummaryWriter(log_dir=log_path, flush_secs=30,
                           purge_step=global_step) # tb refers to tensorboard
    evaluator = SpaceEval(cfg, tb_writer)
    _, logs = evaluator.apply_model(dataset, cfg.device, model, global_step = None, use_global_step=False)
    # ClusteringEval
    clustering_eval = ClusteringEval(cfg, "dummy")
    data, _, _= clustering_eval.collect_z_what_data(logs, dataset, global_step = None)
    relevant_labels, test_x, test_y, train_x, train_y = data[data_category]
    return relevant_labels, test_x, test_y, train_x, train_y

def collect_data_using_files(cfg, data_category="all"):
    print("Collecting data using files")
    game = cfg.exp_name
    labels = np.load(f"labeled/{game}/labels_{data_category}_test.npy")
    z_whats = np.load(f"labeled/{game}/z_whats_{data_category}_test.npy")
    image_refs = pd.read_csv(f"labeled/{game}/image_refs_{data_category}_test.csv")
    image_refs = image_refs.iloc[:, 0].tolist()
    return prepare_data(torch.tensor(z_whats), torch.tensor(labels)), image_refs

def prepare_data(z_what, labels,):
    c = Counter(labels.tolist() if labels is not None else [])
    # Initialization stuff
    relevant_labels = list(c.keys())
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



    
