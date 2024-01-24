
import numpy as np
import os
import os.path as osp
import torch
from sklearn.decomposition import PCA
from PIL import Image
from dataset.z_what import Atari_Z_What
from torch.utils.data import DataLoader
from dataset.atari_data_collector import AtariDataCollector
from pyclustering.cluster import cluster_visualizer
#from pyclustering.cluster.xmeans import xmeans, splitting_type
#from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
#from pyclustering.utils import read_sample
#from pyclustering.samples.definitions import SIMPLE_SAMPLES

from dataset import get_label_list
import numpy as np
from termcolor import colored
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def visualize_classifier(cfg, dataset_mode, data_subset_mode, z_whats, labels, clusters, centers, clf, centroid_labels, relevant_labels, only_collect_first_image_of_consecutive_frames):
    
    # folder creation
    base_folder_path = f"{cfg.resume_ckpt.rsplit('/', 1)[0]}/classifier_visualization"
    os.makedirs(base_folder_path, exist_ok=True)

    # Visualize clustering results
    dim_red_file_name = f"{clf.__class__.__name__}_{data_subset_mode}_clusters_PCA"
    y_pred = clf.predict(z_whats)
    ZWhatPlotter(cfg, base_folder_path, dim_red_file_name).visualize_z_what(z_whats, labels, y_pred, centers, centroid_labels, relevant_labels)

    # Visualize false predictions
    visualize_false_predictions(cfg, dataset_mode, data_subset_mode, labels, centroid_labels, y_pred, only_collect_first_image_of_consecutive_frames)

    # Visualize the selected bbox for each cluster
    images = AtariDataCollector.collect_images(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames)
    pred_boxes = AtariDataCollector.collect_pred_boxes(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames)
    z_whats, labels = AtariDataCollector.collect_z_what_data(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames)
    create_cluster_folders(clf, images, pred_boxes, z_whats, labels, base_folder_path)
    create_one_grid_image_for_each_cluster(base_folder_path)

    ### OLD CODE ###
    # Visualize clustering results
    #sample, centers, dim_name = perform_dimensionality_reduction(z_whats, centers)
    #visualizer = cluster_visualizer()
    #visualizer.append_clusters(clusters, sample)
    #visualizer.append_cluster(centers, None, marker='*', markersize=10)
    #visualizer.save(osp.join(base_path, f"{clf.__class__.__name__}_{data_subset_mode}_clusters_{dim_name}.png"))
    ### OLD CODE END ###

def visualize_false_predictions(cfg, dataset_mode, data_subset_mode, labels, centroid_labels, y_pred, only_collect_first_image_of_consecutive_frames):
     # Visualize false predictions
    images = AtariDataCollector.collect_images(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames)
    pred_boxes = AtariDataCollector.collect_pred_boxes(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames)
    cat_pred_boxes = torch.cat(pred_boxes, dim=0)
    
    image_ref_indexes = []
    for i in range(len(pred_boxes)):
        for j in range(len(pred_boxes[i])):
            image_ref_indexes.append(i)

    base_path = f"{cfg.resume_ckpt.rsplit('/', 1)[0]}/classifier_visualization"
    folder = "wrong_predictions"
    y_pred = [centroid_labels[y_pred_i] for y_pred_i in y_pred]
    os.makedirs(osp.join(base_path, folder), exist_ok=True)
    for i, (y_pred_i, y_gt_i, pred_box_i, image_ref_index) in enumerate(zip(y_pred, labels, cat_pred_boxes, image_ref_indexes)):
        # store examples of all wrong predictions
        if y_pred_i != y_gt_i.item():
            image = images[image_ref_index]
            img = get_bbox_patch_of_image(image, pred_box_i)
            img.save(osp.join(base_path, folder, f"pred_{y_pred_i}_gt_{y_gt_i}_{i}.png"))


def get_bbox_patch_of_image(img, pred_box):
    img = img.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img * 255
    img = img.astype(np.uint8)
    pred_box = pred_box.detach().cpu().numpy()
    pred_box = pred_box*128
    pred_box = np.clip(pred_box, 0, 127) #protect against negative values
    pred_box = pred_box.astype(np.uint8)  
    img = img[pred_box[0]:pred_box[1], pred_box[2]:pred_box[3]]
    img = Image.fromarray(img)
    return img

#def perform_dimensionality_reduction(z_what, centroids,):
#    # perform PCA or TSNE
#    pca = PCA(n_components=2)
#    z_what_emb = pca.fit_transform(z_what)
#    centroid_emb = pca.transform(np.array(centroids))
#    dim_name = "PCA"
#    return z_what_emb, centroid_emb, dim_name

def create_cluster_folders(clf, images, pred_boxes, z_whats, labels, base_path):
    for i, (imgs, pred_boxes, z_what, label) in enumerate(zip(images, pred_boxes, z_whats, labels)):
        z_what = z_what.cpu()
        pred_labels = clf.predict(z_what)
        # cut out the relevant bbox for each pred_box
        for j, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
            # cut out the relevant bbox
            img = get_bbox_patch_of_image(imgs, pred_box)
            # save the bbox
            folder = f"cluster_{pred_label}"
            os.makedirs(osp.join(base_path, folder), exist_ok=True)
            img.save(osp.join(base_path, folder, f"{i}_{j}.png"))
        if i == 100:
            break


def create_one_grid_image_for_each_cluster(base_path):
    # get all cluster folders: check whether is directory and starts with "cluster_"
    cluster_folders = [osp.join(base_path, folder) for folder in os.listdir(base_path) if osp.isdir(osp.join(base_path, folder)) and folder.startswith("cluster_")]
    # open each folder iteratively
    for folder in cluster_folders:
        x_size, y_size = (20,20)
        # create grid image containing each image in the folder
        image_files = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]

        if not image_files:
            print(f"No images found in {folder}")
            continue

        # Create a new blank image to accommodate all the small images
        grid_size = (int(np.sqrt(len(image_files))), int(np.ceil(len(image_files) / np.sqrt(len(image_files)))))
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



class ZWhatPlotter:

    NR_OF_DIMS = 2
    def __init__(self, cfg, folder, file_name, method="pca"):
        print("Initializing ZWhatPlotter with method", method)
        self.cfg = cfg
        self.folder = folder
        self.dim_red_path = f"{self.folder}/{file_name}"
        self.method = method
        self.DISPLAY_CENTROIDS = True
        self.COLORS = ['black', 'r', 'g', 'b', 'c', 'm', 'y', 'pink', 'purple', 'orange',
            'olive', 'brown', 'tomato', 'darkviolet', 'grey', 'chocolate']
        self.edgecolors = False #True iff the ground truth labels and the predicted labels ''(Mixture of some greedy policy and NN) should be drawn in the same image
        self.annotate_wrong_preds = False # True iff the wrong predictions should be annotated in the visualization with the corresponding index of the z_what encoding (only works if edgecolors=True)

    def visualize_z_what(self, z_what, labels, pred_y, centroids, centroid_labels, relevant_labels):
        z_what_emb, centroid_emb, dim_name = self.perform_dimensionality_reduction(z_what, centroids) # either PCA or TSNE

        train_all = torch.cat((z_what, labels.unsqueeze(1)), 1)
        # sort the indices
        sorted_indices = self.get_indices_for_each_label(train_all, relevant_labels)
        # get the label list
        label_list = get_label_list(self.cfg)
        # PLOT
        if self.edgecolors:
            self.edgecolor_visualization(len(z_what), relevant_labels, label_list, pred_y, centroid_labels, sorted_indices, z_what_emb, centroid_emb, dim_name)
        else:
            self.non_edgecolor_visualization(relevant_labels, label_list, pred_y, centroid_labels, sorted_indices, z_what_emb, centroid_emb, dim_name)

    @staticmethod
    def get_indices_for_each_label(train_all, relevant_labels):
        sorted_indices = []
        for i in relevant_labels:
            mask = train_all.T[-1] == i
            indices = torch.nonzero(mask)
            sorted_indices.append(indices)
        return sorted_indices

    def perform_dimensionality_reduction(self, z_what, centroids):
        # perform PCA or TSNE
        if self.method.lower() == "pca":
            print("Running PCA...")
            pca = PCA(n_components=self.NR_OF_DIMS)
            z_what_emb = pca.fit_transform(z_what.numpy())
            centroid_emb = pca.transform(centroids)
            dim_name = "PCA"
        else:
            print("Running t-SNE...")
            print ("If too slow and GPU available, install cuml/MulticoreTSNE (requires conda)")
            tsne = TSNE(n_jobs=4, n_components=self.NR_OF_DIMS, verbose=True, perplexity=min(30, len(z_what)-1, len(centroids)-1))
            z_what_emb = tsne.fit_transform(z_what.numpy())
            centroid_emb = tsne.fit_transform(centroids)
            dim_name = "t-SNE"
        return z_what_emb, centroid_emb, dim_name

    def non_edgecolor_visualization(self, relevant_labels, label_list, y_pred, centroid_label, sorted_indices, z_what_emb, centroid_emb, dim_name):
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
        for i, idx in enumerate(sorted_indices):
            # dimension issue only if there is exactly one object of one kind
            if torch.numel(idx) == 0:
                continue
            y_idx = y_pred[idx] if torch.numel(idx) > 1 else [[y_pred[idx]]]
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
        if self.DISPLAY_CENTROIDS:
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
        plt.tight_layout()

        self.save_plot(fig)

    def edgecolor_visualization(self, n, relevant_labels, label_list, y_pred, centroid_labels, sorted_indices, z_what_emb, centroid_emb, dim_name):
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8, 8)
        ax.set_title(f"Labeled {self.method} of z_whats\n(Inner = GT label, Outer = pred label)")
        ax.set_facecolor((81/255, 89/255, 99/255, 0.4))
        ax.set_xlabel(f"{dim_name} 1", fontsize=20)
        ax.set_ylabel(f"{dim_name} 2", fontsize=20)
        n = min(n, 10000)
        for i, idx in enumerate(sorted_indices):
            if torch.numel(idx) == 0:
                continue
            y_idx = y_pred[idx] if torch.numel(idx) > 1 else [[y_pred[idx]]]
            obj_name = relevant_labels[i]
            colr = self.COLORS[obj_name]
            edge_colors = [self.COLORS[centroid_labels[assign[0]]] for assign in y_idx]
            ax.scatter(z_what_emb[:, 0][idx].squeeze()[:n],
                           z_what_emb[:, 1][idx].squeeze()[:n],
                           c=colr,
                           label=label_list[obj_name],
                           alpha=0.7, edgecolors=edge_colors, s=100, linewidths=2)

            if self.annotate_wrong_preds:
                # annotate all the points where edgecolors are different from colr
                for j, txt in enumerate(idx):
                    if edge_colors[j] != colr:
                        ax.annotate(txt.item(), (z_what_emb[:, 0][idx].squeeze()[j], z_what_emb[:, 1][idx].squeeze()[j]))            
        if self.DISPLAY_CENTROIDS:
            for c_emb, cl in zip(centroid_emb, centroid_labels):
                colr = self.COLORS[cl]
                ax.scatter([c_emb[0]], [c_emb[1]],  c=colr, edgecolors='black', s=100, linewidths=2)
        
        ax.legend(prop={'size': 20})
        plt.tight_layout()

        self.save_plot(fig)

    def save_plot(self, fig):
        directory = f"{self.folder}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f"{self.dim_red_path}.svg")
        plt.savefig(f"{self.dim_red_path}.png")
        print(colored(f"Saved {self.method} images in {self.dim_red_path}", "blue"))
        plt.close(fig)


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