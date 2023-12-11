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
from eval.utils import plot_confusion_matrix
from model.z_what_classifier.z_what_classification import ZWhatClassifierCreator


class ZWhatEvaluator:
    def __init__(self, cfg, title="", method="PCA", indices=None):
        print("Initializing ZWhatEvaluator with method", method)
        self.cfg = cfg
        self.title = title
        self.folder = f'{cfg.logdir}/{cfg.exp_name}'
        self.dim_red_path = f"{self.folder}/{method}{indices if indices else ''}_{title}_{cfg.arch_type}_s{cfg.seed}"
        self.z_what_plotter = ZWhatPlotter(cfg, self.folder, self.dim_red_path, method)
    
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
        # Create K-means
        k_means = ZWhatClassifierCreator(self.cfg).create_k_means(train_x, relevant_labels)
        # Create NN classifier
        nn_class, centroids, centroid_label= ZWhatClassifierCreator(self.cfg).nn_clf_based_on_k_means_centroids(k_means, train_x, train_y, relevant_labels)

        # Eval
        few_shot_accuracy = ZWhatClassifierEvaluator(self.cfg).evaluate_ridge_classifiers_few_shot_accuracy(test_x, test_y, ridge_classifers)
        #mutual_info_scores, train_pred_y = ZWhatClassifierEvaluator(self.cfg).eval_k_means(train_x, train_y, k_means)
        mutual_info_scores, test_pred_y = ZWhatClassifierEvaluator(self.cfg).eval_k_means(test_x, test_y, k_means)
        few_shot_accuracy_cluster_nn = ZWhatClassifierEvaluator(self.cfg).nn_eval(nn_class, test_x, test_y)
        few_shot_accuracy = {**few_shot_accuracy, **few_shot_accuracy_cluster_nn}

        # Confusion matrix
        cm, label_list = self.compute_confusion_matrix(nn_class, test_x, test_y) #TODO: check which classifier to use
        plot_confusion_matrix(cm, label_list, path = f"{self.folder}/confusion_matrix_z_what_{self.title}.png")

        # Visualize
        #self.z_what_plotter.visualize_z_what(train_x, train_y, train_pred_y, centroids, centroid_label, relevant_labels)
        self.z_what_plotter.visualize_z_what(test_x, test_y, test_pred_y, centroids, centroid_label, relevant_labels)

        return mutual_info_scores, self.dim_red_path, few_shot_accuracy

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

class ZWhatPlotter:

    def __init__(self, cfg, folder, dim_red_path, method="pca"):
        print("Initializing ZWhatPlotter with method", method)
        self.cfg = cfg
        self.folder = folder
        self.dim_red_path = dim_red_path
        self.method = method
        self.DISPLAY_CENTROIDS = True
        self.COLORS = ['black', 'r', 'g', 'b', 'c', 'm', 'y', 'pink', 'purple', 'orange',
            'olive', 'brown', 'tomato', 'darkviolet', 'grey', 'chocolate']
        self.edgecolors = False #True iff the ground truth labels and the predicted labels ''(Mixture of some greedy policy and NN) should be drawn in the same image
        self.dim = 2 # Number of dimension for PCA/TSNE visualization

    def visualize_z_what(self, z_what, labels, y, centroids, centroid_label, relevant_labels):
        z_what_emb, centroid_emb, dim_name = self.perform_dimensionality_reduction(z_what, centroids) # either PCA or TSNE

        train_all = torch.cat((z_what, labels.unsqueeze(1)), 1)
        # sort the indices
        sorted = []
        for i in relevant_labels:
            mask = train_all.T[-1] == i
            indices = torch.nonzero(mask)
            sorted.append(indices)
        # get the label list
        label_list = get_label_list(self.cfg)
        # PLOT
        if self.edgecolors:
            self.edgecolor_visualization(len(z_what), relevant_labels, label_list, y, centroid_label, sorted, z_what_emb, centroid_emb)
        else:
            self.non_edgecolor_visualization(relevant_labels, label_list, y, centroid_label, sorted, z_what_emb, centroid_emb, dim_name)

    def perform_dimensionality_reduction(self, z_what, centroids):
        # perform PCA or TSNE
        if self.method.lower() == "pca":
            print("Running PCA...")
            pca = PCA(n_components=self.dim)
            z_what_emb = pca.fit_transform(z_what.numpy())
            centroid_emb = pca.transform(centroids)
            dim_name = "PCA"
        else:
            print("Running t-SNE...")
            print ("If too slow and GPU available, install cuml/MulticoreTSNE (requires conda)")
            tsne = TSNE(n_jobs=4, n_components=self.dim, verbose=True, perplexity=min(30, len(z_what)-1, len(centroids)-1))
            z_what_emb = tsne.fit_transform(z_what.numpy())
            centroid_emb = tsne.fit_transform(centroids)
            dim_name = "t-SNE"
        return z_what_emb, centroid_emb, dim_name

    def non_edgecolor_visualization(self, relevant_labels, label_list, y, centroid_label, sorted, z_what_emb, centroid_emb, dim_name):
        fig, axs = plt.subplots(2, 1)
        fig.set_size_inches(8, 15)
        axs[0].set_title("Ground Truth Labels", fontsize=20)
        axs[1].set_title("Labels Following Clustering", fontsize=20)
        for ax in axs:
            ax.set_facecolor((81/255, 89/255, 99/255, 0.4))
            ax.set_xlabel(f"{dim_name} 1", fontsize=20)
            ax.set_ylabel(f"{dim_name} 2", fontsize=20)
        all_colors = []
        all_edge_colors = []
        for i, idx in enumerate(sorted):
                # dimension issue only if there is exactly one object of one kind
            if torch.numel(idx) == 0:
                continue
            y_idx = y[idx] if torch.numel(idx) > 1 else [[y[idx]]]
            obj_name = relevant_labels[i]
            colr = self.COLORS[obj_name]
            edge_colors = [self.COLORS[centroid_label[assign[0]]] for assign in y_idx]
            all_edge_colors.extend(edge_colors)
            all_colors.append(colr)
            axs[0].scatter(z_what_emb[:, 0][idx].squeeze(),
                               z_what_emb[:, 1][idx].squeeze(),
                               c=colr,
                               label=label_list[obj_name],
                               alpha=0.7)
            axs[1].scatter(z_what_emb[:, 0][idx].squeeze(),
                               z_what_emb[:, 1][idx].squeeze(),
                               c=edge_colors,
                               alpha=0.7)
        print(all_colors)
        print(set(all_edge_colors))
        for c_emb, cl in zip(centroid_emb, centroid_label):
            colr = self.COLORS[cl]
            axs[0].scatter([c_emb[0]],
                               [c_emb[1]],
                               c=colr,
                               edgecolors='black', s=100, linewidths=2)
            axs[1].scatter([c_emb[0]],
                               [c_emb[1]],
                               c=colr,
                               edgecolors='black', s=100, linewidths=2)

        axs[0].legend(prop={'size': 20})
            # axs[1].legend(prop={'size': 17})
        if not os.path.exists(f"{self.folder}"):
            os.makedirs(f"{self.folder}")
            # fig.suptitle(f"Embeddings of {arch}", fontsize=20)
        plt.tight_layout()
            # plt.subplots_adjust(top=0.65)
        plt.savefig(f"{self.dim_red_path}.svg")
        plt.savefig(f"{self.dim_red_path}.png")
        print(colored(f"Saved {self.method} images in {self.dim_red_path}", "blue"))
        plt.close(fig)

    @staticmethod
    def edgecolor_visualization(self, n, relevant_labels, label_list, y, centroid_label, sorted, z_what_emb, centroid_emb):
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(20, 8)
        ax.set_facecolor((0.3, 0.3, 0.3))
        plt.suptitle(f"Labeled {self.method} of z_whats", y=0.96, fontsize=28)
        plt.title("Inner Color is GT, Outer is greedy Centroid-based label", fontsize=18, pad=20)
        n = min(n, 10000)
        for i, idx in enumerate(sorted):
            if torch.numel(idx) == 0:
                continue
            y_idx = y[idx] if torch.numel(idx) > 1 else [[y[idx]]]
            colr = self.COLORS[relevant_labels[i]]
            edge_colors = [self.COLORS[centroid_label[assign[0]]] for assign in y_idx]
            ax.scatter(z_what_emb[:, 0][idx].squeeze()[:1],
                           z_what_emb[:, 1][idx].squeeze()[:1],
                           c=colr, label=label_list[relevant_labels[i]].replace("_", " "),
                           alpha=0.7)
            ax.scatter(z_what_emb[:, 0][idx].squeeze()[:n],
                           z_what_emb[:, 1][idx].squeeze()[:n],
                           c=colr,
                           alpha=0.7, edgecolors=edge_colors, s=100, linewidths=2)

        if self.DISPLAY_CENTROIDS:
            for c_emb, cl in zip(centroid_emb, centroid_label):
                colr = self.COLORS[cl]
                ax.scatter([c_emb[0]], [c_emb[1]],  c=colr, edgecolors='black', s=100, linewidths=2)
        plt.legend(prop={'size': 6})
        directory = f"{self.folder}"
        if not os.path.exists(directory):
            print(f"Writing {self.method} to {directory}")
            os.makedirs(directory)
        plt.savefig(f"{self.dim_red_path}.svg")
        plt.savefig(f"{self.dim_red_path}.png")

    def no_z_whats_plots(self):
        fig, axs = plt.subplots(2, 1)
        fig.set_size_inches(8, 15)
        axs[0].set_title("Ground Truth Labels", fontsize=20)
        axs[1].set_title("Labels Following Clustering", fontsize=20)
        s = "No z_what extracted\n      by the model"
        dim_name = "PCA" if self.method == "pca" else "t-SNE"
        for ax in axs:
            ax.set_xlabel(f"{dim_name} 1", fontsize=20)
            ax.set_ylabel(f"{dim_name} 2", fontsize=20)
            ax.text(0.03, 0.1, s, rotation=45, fontsize=45)
        plt.tight_layout()

        if not os.path.exists(f"{self.folder}"):
            os.makedirs(f"{self.folder}")
        plt.savefig(f"{self.dim_red_path}.svg")
        plt.savefig(f"{self.dim_red_path}.png")
        print(colored(f"Saved empty {self.method} images in {self.dim_red_path}", "red"))
        plt.close(fig)

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