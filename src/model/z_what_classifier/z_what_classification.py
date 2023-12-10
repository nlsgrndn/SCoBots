import os
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture

import joblib

class ZWhatClassifierCreator:

    few_shot_values = [1, 4, 16, 64]
    N_NEIGHBORS = 24

    def __init__(self, cfg):
        self.cfg = cfg
        os.makedirs(f'{self.cfg.logdir}/{self.cfg.exp_name}', exist_ok=True)
        os.makedirs(f'classifiers', exist_ok=True)

    def create_ridge_classifiers(self, relevant_labels, train_x, train_y):
        # separate the data by class label
        z_what_by_class_label = {rl: train_x[train_y == rl] for rl in relevant_labels}
        labels_by_class_label = {rl: train_y[train_y == rl] for rl in relevant_labels}

        classifiers = {}
        for training_objects_per_class in ZWhatClassifierCreator.few_shot_values:
            current_train_sample = torch.cat([z_what_by_class_label[rl][:training_objects_per_class] for rl in relevant_labels])
            current_train_labels = torch.cat([labels_by_class_label[rl][:training_objects_per_class] for rl in relevant_labels])
            clf = RidgeClassifier()
            clf.fit(current_train_sample, current_train_labels)
            classifiers[training_objects_per_class] = clf
            filename = f'{self.cfg.logdir}/{self.cfg.exp_name}/z_what-classifier_with_{training_objects_per_class}.joblib.pkl'
            joblib.dump(clf, filename)
            #model_name = self.cfg.resume_ckpt.split("/")[-1].replace(".pth", "")

        clf = classifiers[max(ZWhatClassifierCreator.few_shot_values)] 
        model_name = self.cfg.exp_name
        save_path = f"classifiers/{model_name}_z_what_classifier.joblib.pkl"
        joblib.dump(clf, save_path)
        print(f"Saved classifiers in {save_path}")
        return classifiers

    def create_k_means(self, z_what, relevant_labels):
        k_means = KMeans(n_clusters=len(relevant_labels))
        k_means.fit(z_what)
        return k_means


    # high level: essentially assign semantic labels to k_means centroids instead of just enumerating the clusters
    def nn_clf_based_on_k_means_centroids(self, k_means, train_x, train_y, relevant_labels):
        # Assign a label to each cluster centroid
        centroids = k_means.cluster_centers_
        X = train_x.numpy()
        n_neighbors = min(self.N_NEIGHBORS, len(X))
        nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
        _, z_w_idx = nn.kneighbors(centroids)
        centroid_label = []
        for cent, nei in zip(centroids, z_w_idx):
            count = {rl: 0 for rl in relevant_labels}
            added = False
            for i in range(n_neighbors):
                nei_label = train_y[nei[i]].item()
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


    
