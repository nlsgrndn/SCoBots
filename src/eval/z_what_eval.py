import argparse
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import warnings
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from argparse import Namespace
import os
from sklearn.neighbors import KNeighborsClassifier
from dataset import get_label_list
from collections import Counter
import numpy as np
import joblib
from termcolor import colored
from sklearn.manifold import TSNE
from  space_models.z_what_classifier.z_what_classification import ZWhatClassifierCreator
from classifier_visualization.classifier_vis import ZWhatPlotter


class ZWhatEvaluator:
    def __init__(self, cfg, title="", method="PCA", indices=None):
        print("Initializing ZWhatEvaluator with method", method)
        self.cfg = cfg
        self.data_subset_mode = title
        self.folder = f'{cfg.logdir}/{cfg.exp_name}/encoding_eval'
        os.makedirs(f'{self.folder}', exist_ok=True)
        self.method = method

    def evaluate_z_what(self, train_x, train_y, test_x, test_y, relevant_labels):
        """
        This function evaluates the z_what encoding using PCA or t-SNE and plots the results.
        :param train_x: (#objects, encoding_dim)
        :param train_y: (#objects)
        :param test_x: (#objects, encoding_dim)
        :param test_y: (#objects)
        :param relevant_labels: list of relevant labels
        :return:
            result: metrics
            path: to pca or tsne image
            accuracy: few shot accuracy
        """
        if train_x is None: # representative for all inputs
            self.z_what_plotter.no_z_whats_plots()
            return None, None, None
        
        # Create ridge classifiers
        ridge_classifers = ZWhatClassifierCreator(self.cfg).create_ridge_classifiers(relevant_labels, train_x, train_y)
        for data_points, ridge_clf in ridge_classifers.items():
            ZWhatClassifierCreator(self.cfg).save_classifier(
                clf=ridge_clf,
                clf_name=f"ridge_clf_{data_points}",
                folder=os.path.join(self.folder, "classifiers"),
                data_subset_mode=self.data_subset_mode,
            )

        # Create K-means
        k_means = ZWhatClassifierCreator(self.cfg).create_k_means(train_x, relevant_labels)
        
        # Create NN classifier based on K-means
        nn_class, centroids, centroid_label= ZWhatClassifierCreator(self.cfg).nn_clf_based_on_k_means_centroids(k_means, train_x, train_y, relevant_labels)
        ZWhatClassifierCreator(self.cfg).save_classifier(
            clf = nn_class,
            clf_name=f"nn_clf",
            folder=os.path.join(self.folder, "classifiers"),
            data_subset_mode=self.data_subset_mode,
        )

        # Eval
        few_shot_accuracy = ZWhatClassifierEvaluator(self.cfg).evaluate_ridge_classifiers_few_shot_accuracy(test_x, test_y, ridge_classifers)
        _, train_pred_y = ZWhatClassifierEvaluator(self.cfg).eval_k_means(train_x, train_y, k_means) #just for visualization
        mutual_info_scores, test_pred_y = ZWhatClassifierEvaluator(self.cfg).eval_k_means(test_x, test_y, k_means)
        few_shot_accuracy_cluster_nn = ZWhatClassifierEvaluator(self.cfg).nn_eval(nn_class, test_x, test_y)
        few_shot_accuracy = {**few_shot_accuracy, **few_shot_accuracy_cluster_nn}


        
        # Confusion matrix
        #cm, label_list = self.compute_confusion_matrix(nn_class, test_x, test_y) #TODO: check which classifier to use
        #path = f"{self.folder}/confusion_matrix_z_what_{self.data_subset_mode}.png"
        #plot_confusion_matrix(cm, label_list, path) 

        
        # Visualize
        dim_red_file_name = f"encoding_space_{self.data_subset_mode}_{self.method}"
        ZWhatPlotter(self.cfg, self.folder, dim_red_file_name + "_trainsplit", self.method).visualize_z_what(train_x, train_y, train_pred_y, centroids, centroid_label, relevant_labels)
        ZWhatPlotter(self.cfg, self.folder, dim_red_file_name + "_testsplit", self.method).visualize_z_what(test_x, test_y, test_pred_y, centroids, centroid_label, relevant_labels)

        result_file_path = f"{self.folder}/{dim_red_file_name}_trainsplit.png" # is used for logging

        return mutual_info_scores, result_file_path, few_shot_accuracy

    def compute_confusion_matrix(self, clf, inputs, labels):
        """
        Compute the confusion matrix for a given classifier
        :param clf: classifier
        :param inputs: inputs
        :param labels: labels
        :return: confusion matrix
        """
        y_pred = clf.predict(inputs)
        labels = labels.numpy()
        label_list_idx = list(Counter(np.concatenate((labels, y_pred))).keys())
        label_list = [get_label_list(self.cfg)[i] for i in label_list_idx]
        cm = metrics.confusion_matrix(labels, y_pred, labels=label_list_idx)
        return cm, label_list

class ZWhatClassifierEvaluator:

    few_shot_values = [1, 4, 16, 64]
    def __init__(self, cfg):
        self.cfg = cfg
        os.makedirs(f'{self.cfg.logdir}/{self.cfg.exp_name}', exist_ok=True)
        os.makedirs(f'classifiers', exist_ok=True)

    def evaluate_ridge_classifiers_few_shot_accuracy(self, test_x, test_y, classifiers):
        few_shot_accuracy = {}
        for training_objects_per_class in ZWhatClassifierEvaluator.few_shot_values:
            clf = classifiers[training_objects_per_class]
            acc = clf.score(test_x, test_y)
            few_shot_accuracy[f'few_shot_accuracy_with_{training_objects_per_class}'] = acc
        return few_shot_accuracy

    def eval_k_means(self,z_what, labels, k_means):
        y =k_means.predict(z_what)
        results = {
            'adjusted_mutual_info_score': metrics.adjusted_mutual_info_score(labels, y),
            'adjusted_rand_score': metrics.adjusted_rand_score(labels, y),
        }
        return results, y

    def nn_eval(self, nn_class, test_x, test_y):
        few_shot_accuracy_cluster_nn = {'few_shot_accuracy_cluster_nn' : nn_class.score(test_x, test_y)}
        return few_shot_accuracy_cluster_nn