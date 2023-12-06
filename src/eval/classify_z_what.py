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


class ZWhatEvaluator:
    def __init__(self, cfg, title="", method="pca", indices=None, edgecolors=False, dim=2):
        print("Initializing ZWhatEvaluator with method", method)
        self.cfg = cfg
        self.folder = f'{cfg.logdir}/{cfg.exp_name}'
        self.dim_red_path = f"{self.folder}/{method}{indices if indices else ''}_{title}_{cfg.arch_type}_s{cfg.seed}"
        self.method = method
        self.indices = indices # 'The relevant objects by their index, e.g. \"0,1\" for Pacman and Sue')
        self.N_NEIGHBORS = 24
        self.DISPLAY_CENTROIDS = True
        self.COLORS = ['black', 'r', 'g', 'b', 'c', 'm', 'y', 'pink', 'purple', 'orange',
            'olive', 'brown', 'tomato', 'darkviolet', 'grey', 'chocolate']
        self.edgecolors = edgecolors
        self.dim = dim

    def prepare_data(self, z_what, labels,):
        c = Counter(labels.tolist() if labels is not None else [])
        if self.cfg.train.log:
            print("Distribution of matched labels:", c)
        # Initialization stuff
        relevant_labels = [int(part) for part in self.indices.split(',')] if self.indices else list(c.keys())

        # Filter out the irrelevant labels
        z_what, labels = self.only_keep_relevant_data(z_what, labels, relevant_labels)
        # Split the data into train and test
        train_x, train_y, test_x, test_y = self.train_test_split(z_what, labels, train_portion=0.9)

        if len(c) < 2 or len(torch.unique(train_y)) < 2:
            return None

        # Set up dictionaries to store the z_what and labels for each game
        z_what_by_game = {rl: train_x[train_y == rl] for rl in relevant_labels}
        labels_by_game = {rl: train_y[train_y == rl] for rl in relevant_labels}

        return relevant_labels, z_what_by_game, labels_by_game, test_x, test_y, train_x, train_y

    def evaluate_z_what(self, z_what, labels,):
        """
        This function evaluates the z_what encoding using PCA or t-SNE and plots the results.
        :param z_what: (#objects, encoding_dim)
        :param labels: (#objects)
        :return:
            result: metrics
            path: to pca or tsne image
            accuracy: few shot accuracy
        """
        # Prepare data

        data = self.prepare_data(z_what, labels)
        if data is None:
            self.no_z_whats_plots()
            return Counter(), self.dim_red_path, Counter()
        else:
            relevant_labels, z_what_by_game, labels_by_game, test_x, test_y, train_x, train_y = data
        
        # Create classifiers
        ridge_classifers = ZWhatClassifierCreator(self.cfg).create_ridge_classifiers(relevant_labels, z_what_by_game, labels_by_game)
        # Create K-means
        k_means = ZWhatClassifierCreator(self.cfg).create_k_means(z_what, relevant_labels)
        # Create NN classifier
        nn_class, centroids, centroid_label= ZWhatClassifierCreator(self.cfg).nn_clf_based_on_k_means_centroids(k_means, labels, relevant_labels, train_x, self.N_NEIGHBORS)

        # Eval
        few_shot_accuracy = ZWhatClassifierCreator(self.cfg).evaluate_ridge_classifiers_few_shot_accuracy(test_x, test_y, ridge_classifers)
        mutual_info_scores, y = ZWhatClassifierCreator(self.cfg).eval_k_means(z_what, labels, k_means)
        few_shot_accuracy_cluster_nn = ZWhatClassifierCreator(self.cfg).nn_eval(nn_class, test_x, test_y)
        few_shot_accuracy = {**few_shot_accuracy, **few_shot_accuracy_cluster_nn}

        # Visualize
        self.visualize_z_what(z_what, labels, y, centroids, centroid_label, relevant_labels)

        return mutual_info_scores, self.dim_red_path, few_shot_accuracy
    
    @staticmethod
    def train_test_split(z_what, labels, train_portion=0.9):
        nb_sample = int(train_portion * len(labels))
        train_x = z_what[:nb_sample]
        train_y = labels[:nb_sample]
        test_x = z_what[nb_sample:]
        test_y = labels[nb_sample:]
        return train_x, train_y, test_x, test_y

    @staticmethod
    def only_keep_relevant_data(z_what, labels, relevant_labels):
        relevant = torch.zeros(labels.shape, dtype=torch.bool)
        for rl in relevant_labels:
            relevant |= labels == rl
        return z_what[relevant], labels[relevant]

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

    def non_edgecolor_visualization(self,relevant_labels, label_list, y, centroid_label, sorted, z_what_emb, centroid_emb, dim_name):
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

class ZWhatClassifierCreator:

    few_shot_values = [1, 4, 16, 64]
    def __init__(self, cfg):
        self.cfg = cfg
        os.makedirs(f'{self.cfg.logdir}/{self.cfg.exp_name}', exist_ok=True)
        os.makedirs(f'classifiers', exist_ok=True)

    def create_ridge_classifiers(self, relevant_labels, z_what_by_game, labels_by_game):
        classifiers = {}
        for training_objects_per_class in ZWhatClassifierCreator.few_shot_values:
            current_train_sample = torch.cat([z_what_by_game[rl][:training_objects_per_class] for rl in relevant_labels])
            current_train_labels = torch.cat([labels_by_game[rl][:training_objects_per_class] for rl in relevant_labels])
            clf = RidgeClassifier()
            clf.fit(current_train_sample, current_train_labels)
            classifiers[training_objects_per_class] = clf
            filename = f'{self.cfg.logdir}/{self.cfg.exp_name}/z_what-classifier_with_{training_objects_per_class}.joblib.pkl'
            joblib.dump(clf, filename)
            model_name = self.cfg.resume_ckpt.split("/")[-1].replace(".pth", "")
            save_path = f"classifiers/{model_name}_z_what_classifier.joblib.pkl"
            joblib.dump(clf, save_path)
            print(f"Saved classifiers in {save_path}")
        return classifiers

    def evaluate_ridge_classifiers_few_shot_accuracy(self, test_x, test_y, classifiers):
        few_shot_accuracy = {}
        for training_objects_per_class in ZWhatClassifierCreator.few_shot_values:
            clf = classifiers[training_objects_per_class]
            acc = clf.score(test_x, test_y)
            few_shot_accuracy[f'few_shot_accuracy_with_{training_objects_per_class}'] = acc
        return few_shot_accuracy

    def create_k_means(self, z_what, relevant_labels):
        k_means = KMeans(n_clusters=len(relevant_labels))
        k_means.fit(z_what)
        return k_means

    def eval_k_means(self,z_what, labels, k_means):
        y =k_means.predict(z_what)
        results = {
            'adjusted_mutual_info_score': metrics.adjusted_mutual_info_score(labels, y),
            'adjusted_rand_score': metrics.adjusted_rand_score(labels, y),
        }
        return results, y

    def nn_clf_based_on_k_means_centroids(self, k_means, labels, relevant_labels, train_x, n_neighbors):
        # Assign a label to each cluster centroid
        centroids = k_means.cluster_centers_
        X = train_x.numpy()
        n_neighbors = min(n_neighbors, len(X))
        nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
        _, z_w_idx = nn.kneighbors(centroids)
        centroid_label = []
        for cent, nei in zip(centroids, z_w_idx):
            count = {rl: 0 for rl in relevant_labels}
            added = False
            for i in range(n_neighbors):
                nei_label = labels[nei[i]].item()
                count[nei_label] += 1
                if count[nei_label] > 6.0 / (i + 1) if nei_label in centroid_label else 3.0 / (i + 1):
                    centroid_label.append(nei_label)
                    added = True
                    break
            if not added:
                leftover_labels = [i for i in relevant_labels if i not in centroid_label]
                centroid_label.append(leftover_labels[0])
        # Train a K-nearest neighbors classifier on the centroids
        nn_class = KNeighborsClassifier(n_neighbors=1)
        nn_class.fit(centroids, centroid_label)
        return nn_class, centroids, centroid_label 

    def nn_eval(self, nn_class, test_x, test_y):
        few_shot_accuracy_cluster_nn = {'few_shot_accuracy_cluster_nn' : nn_class.score(test_x, test_y)}
        return few_shot_accuracy_cluster_nn
    
# all_train_labels = pd.read_csv(f"../aiml_atari_data/rgb/MsPacman-v0/train_labels.csv")
# all_validation_labels = pd.read_csv(f"../aiml_atari_data/rgb/MsPacman-v0/validation_labels.csv")

# label_list = ["pacman", 'sue', 'inky', 'pinky', 'blinky', "blue_ghost",
#               "white_ghost", "fruit", "save_fruit", "life", "life2", "score0"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the z_what encoding')
    parser.add_argument('-edgecolors', type=bool, default=True,
                        help='True iff the ground truth labels and the predicted labels '
                             '(Mixture of some greedy policy and NN) should be drawn in the same image')
    parser.add_argument('-dim', type=int, choices=[2, 3], default=2,
                        help='Number of dimension for PCA/TSNE visualization')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nb_used_sample = 500

    # z_what_train = torch.randn((400, 32))
    # train_labels = torch.randint(high=8, size=(400,))
    #z_what_train = torch.load(f"labeled/z_what_validation.pt")
    #train_labels = torch.load(f"labeled/labels_validation.pt")
    # same but with train
    z_what_train = torch.load(f"labeled/z_what_train.pt")
    train_labels = torch.load(f"labeled/labels_train.pt")
    cfg = None #TODO fix
    z_what_evaluator = ZWhatEvaluator(cfg, method="pca", indices="0,1", edgecolors = args.edgecolors, dim=args.dim)
    z_what_evaluator.evaluate_z_what(cfg, z_what_train, train_labels, nb_used_sample,)
