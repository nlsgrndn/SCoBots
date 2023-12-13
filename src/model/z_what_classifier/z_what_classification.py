import os
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
# import x-means from pyclustering
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster import cluster_visualizer
from sklearn.decomposition import PCA
import numpy as np

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
            classifiers[training_objects_per_class] = self.create_ridge_classifier(current_train_sample, current_train_labels)
        return classifiers
    
    def create_ridge_classifier(self, train_x, train_y):
        clf = RidgeClassifier()
        clf.fit(train_x, train_y)
        return clf

    def create_k_means(self, z_what, relevant_labels):
        k_means = KMeans(n_clusters=len(relevant_labels))
        k_means.fit(z_what)
        return k_means

    def save_classifier(self, clf, model_name):
        folder = f'{self.cfg.logdir}/{self.cfg.exp_name}/{model_name}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = 'z_what-classifier_{type(clf)}.joblib.pkl'
        path = os.path.join(folder, filename)
        joblib.dump(clf, path)


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

    def create_x_means(self, sample, seed):
        amount_initial_centers = 2
        initial_centers = kmeans_plusplus_initializer(sample, amount_initial_centers, random_state=seed).initialize()
        # Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum
        # number of clusters that can be allocated is 20.
        xmeans_instance = xmeans(sample, initial_centers, kmax=20)
        xmeans_instance.process()
        # Extract clustering results: clusters and their centers
        clusters = xmeans_instance.get_clusters()
        centers = xmeans_instance.get_centers()
        # Print total sum of metric errors
        print("Total WCE:", xmeans_instance.get_total_wce())
        # Visualize clustering results
        visualizer = cluster_visualizer()
        sample, centers, dim_name = self.perform_dimensionality_reduction(sample, centers)
        visualizer.append_clusters(clusters, sample)
        visualizer.append_cluster(centers, None, marker='*', markersize=10)
        visualizer.save('xmeans.png')

        return xmeans_instance, centers, clusters

    def perform_dimensionality_reduction(self, z_what, centroids,):
        # perform PCA or TSNE
        print("Running PCA...")
        pca = PCA(n_components=2)
        z_what_emb = pca.fit_transform(z_what)
        centroid_emb = pca.transform(np.array(centroids))
        dim_name = "PCA"
        return z_what_emb, centroid_emb, dim_name

    
