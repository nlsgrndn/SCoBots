
import numpy as np
import os
import os.path as osp
import torch
from sklearn.decomposition import PCA
from PIL import Image
from dataset.z_what import Atari_Z_What
from torch.utils.data import DataLoader

from pyclustering.cluster import cluster_visualizer
#from pyclustering.cluster.xmeans import xmeans, splitting_type
#from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
#from pyclustering.utils import read_sample
#from pyclustering.samples.definitions import SIMPLE_SAMPLES

def visualize_classifier(cfg, dataset_mode, data_subset_mode, train_x, clusters, centers, clf):
    base_path = f"classifier_visualization/{cfg.exp_name}/"
    os.makedirs(base_path, exist_ok=True)

    # Visualize clustering results
    visualizer = cluster_visualizer()
    sample, centers, dim_name = perform_dimensionality_reduction(train_x, centers)
    visualizer.append_clusters(clusters, sample)
    visualizer.append_cluster(centers, None, marker='*', markersize=10)
    visualizer.save(osp.join(base_path, f"{clf.__class__.__name__}_{data_subset_mode}_clusters_{dim_name}.png"))

    # load data for visualization
    images = collect_images_using_dataloader(cfg, dataset_mode, data_subset_mode)
    pred_boxes = collect_pred_boxes_using_dataloader(cfg, dataset_mode, data_subset_mode)
    z_whats, labels = collect_z_what_data_for_data_subset_mode(cfg, dataset_mode, data_subset_mode)
    # Visualize the selected bbox for each cluster
    create_cluster_folders(clf, images, pred_boxes, z_whats, labels, base_path)
    create_one_grid_image_for_each_cluster(base_path)

def perform_dimensionality_reduction(z_what, centroids,):
    # perform PCA or TSNE
    pca = PCA(n_components=2)
    z_what_emb = pca.fit_transform(z_what)
    centroid_emb = pca.transform(np.array(centroids))
    dim_name = "PCA"
    return z_what_emb, centroid_emb, dim_name

def collect_z_what_data_for_data_subset_mode(cfg, dataset_mode, data_subset_mode):        
    atari_z_what_dataset = Atari_Z_What(cfg, dataset_mode, data_subset_mode, return_keys = ["z_whats_pres_s", "gt_labels_for_pred_boxes"])
    atari_z_what_dataloader = DataLoader(atari_z_what_dataset, batch_size=1, shuffle=False, num_workers=0) #batch_size must be 1
    z_whats = []
    labels = []
    for batch in atari_z_what_dataloader:
        curr_z_whats = batch["z_whats_pres_s"]
        curr_labels = batch["gt_labels_for_pred_boxes"]
        z_whats.extend([curr_z_whats[i][0] for i in range(len(curr_z_whats))]) #[0] because batch_size = 1 and dataloader implicitly adds a batch dimension
        labels.extend([curr_labels[i][0] for i in range(len(curr_labels))]) #[0] because batch_size = 1 and dataloader implicitly adds a batch dimension
    return z_whats, labels
        
def collect_images_using_dataloader(cfg, dataset_mode, data_subset_mode):
    atari_z_what_dataset = Atari_Z_What(cfg, dataset_mode, data_subset_mode, return_keys = ["imgs"])
    atari_z_what_dataloader = DataLoader(atari_z_what_dataset, batch_size=1, shuffle=False, num_workers=0) #batch_size must be 1
    images = []
    for batch in atari_z_what_dataloader:
        curr_images = batch["imgs"][0] #[0] because batch_size = 1 and dataloader implicitly adds a batch dimension
        images.extend(curr_images)
    images = torch.stack(images, dim=0)
    return images

def collect_pred_boxes_using_dataloader(cfg, dataset_mode, data_subset_mode):
    atari_z_what_dataset = Atari_Z_What(cfg, dataset_mode, data_subset_mode, return_keys = ["pred_boxes"])
    atari_z_what_dataloader = DataLoader(atari_z_what_dataset, batch_size=1, shuffle=False, num_workers=0) #batch_size must be 1
    pred_boxes = []
    for batch in atari_z_what_dataloader:
            curr_pred_boxes= batch["pred_boxes"] 
            pred_boxes.extend([curr_pred_boxes[i][0] for i in range(len(curr_pred_boxes))]) #[0] because batch_size = 1 and dataloader implicitly adds a batch dimension
    return pred_boxes

def create_cluster_folders(clf, images, pred_boxes, z_whats, labels, base_path):
    for i, (imgs, pred_boxes, z_what, label) in enumerate(zip(images, pred_boxes, z_whats, labels)):
        z_what = z_what.cpu()
        pred_labels = clf.predict(z_what)
        # cut out the relevant bbox for each pred_box
        for j, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
            # cut out the relevant bbox
            img = imgs.detach().cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img * 255
            img = img.astype(np.uint8)
            pred_box = pred_box.detach().cpu().numpy()
            pred_box = pred_box*128
            pred_box = pred_box.astype(np.uint8)
            img = img[pred_box[0]:pred_box[1], pred_box[2]:pred_box[3]]
            img = Image.fromarray(img)
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