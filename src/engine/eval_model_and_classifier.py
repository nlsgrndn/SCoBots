from sklearn.metrics import confusion_matrix
from dataset.atari_labels import label_list_for,  get_moving_indices
import numpy as np
from typing import List, Union
from eval.utils import plot_confusion_matrix
from space_and_moc_utils.bbox_matching import match_bounding_boxes
from dataset.z_what import Atari_Z_What
from torch.utils.data import DataLoader
from engine.utils import load_classifier
from sklearn.metrics import classification_report
import pandas as pd
from create_latent_dataset import create_latent_dataset
from dataset.atari_labels import label_list_for

def eval_model_and_classifier(cfg):
    dataset_mode = "test"
    data_subset_mode = "relevant"
    game = cfg.exp_name.split("_")[0]
    base_path = cfg.resume_ckpt.rsplit('/', 1)[0]
    folder_path = base_path.replace("space_weights", "classifier")
    create_latent_dataset(cfg, dataset_mode)
    dataset = Atari_Z_What(cfg, dataset_mode, boxes_subset=data_subset_mode, return_keys=["gt_bbs_and_labels", "pred_boxes", "z_whats_pres_s"])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    clf, centroid_labels = load_classifier(folder_path=folder_path, clf_name="nn", data_subset_mode=data_subset_mode)
    label_list = label_list_for(game)
    actual_labels_combined, predicted_labels_combined, label_list = get_data(dataloader, clf, centroid_labels, label_list)
    
    print("Evaluating model and classifier on", len(actual_labels_combined), "z_what encodings")

    # compute metrics
    result_dict = classification_report(actual_labels_combined, predicted_labels_combined, labels=np.arange(len(label_list)), target_names=label_list, output_dict=True)
    conf_matrix = confusion_matrix_visualization(actual_labels_combined, predicted_labels_combined, label_list, game)
    recall = compute_recall(actual_labels_combined, predicted_labels_combined, label_list)
    precision = compute_precision(actual_labels_combined, predicted_labels_combined, label_list)
    f1_score = compute_f1_score(recall, precision)

    #save precision, recall in dataframe
    df = pd.DataFrame(columns=["relevant_precision", "relevant_recall", "relevant_f1_score"])
    df.loc[0] = [precision, recall, f1_score]
    df.to_csv(f"{base_path}/eval_model_and_classifier.csv")

def get_data(dataloader, classifier, centroid_labels_dict, label_list: List[Union[str, int]]):
    actual_labels_combined = []
    predicted_labels_combined = []
    for i, data in enumerate(dataloader):
        gt_bbs_and_labels = data["gt_bbs_and_labels"]
        pred_boxes = data["pred_boxes"]
        z_whats_pres_s = data["z_whats_pres_s"]

        for i in range(dataloader.dataset.T):
            curr_pred_boxes, curr_z_whats_pres_s, curr_gt_bbs_and_labels = pred_boxes[i][0], z_whats_pres_s[i][0], gt_bbs_and_labels[i][0]
            curr_pred_boxes, curr_z_whats_pres_s, curr_gt_bbs_and_labels = curr_pred_boxes.to("cpu"), curr_z_whats_pres_s.to("cpu"), curr_gt_bbs_and_labels.to("cpu")
            pred_labels = classifier.predict(curr_z_whats_pres_s)
            if centroid_labels_dict is not None:
                pred_labels = [centroid_labels_dict[label] for label in pred_labels]
            pred_labels = np.array(pred_labels)
            curr_pred_bbs_and_labels = np.concatenate((curr_pred_boxes, pred_labels[:, np.newaxis]), axis=1)
            actual_labels, predicted_labels = match_bounding_boxes(curr_gt_bbs_and_labels, curr_pred_bbs_and_labels, label_list) # len(actual) = len(labels[i]) + # of predicted boxes that are not matched
            actual_labels_combined.extend(actual_labels)
            predicted_labels_combined.extend(predicted_labels)

    label_list.append("not_an_object") # add no_object for case that no bounding box is predicted or predicted bounding box is not matched
    label_list.append("not_detected") # add not_detected for case that no bounding box is predicted or predicted bounding box is not matched
    return actual_labels_combined, predicted_labels_combined, label_list

def confusion_matrix_visualization(actual_labels_combined, predicted_labels_combined, label_list: List[Union[str, int]], game: str):
    cm = confusion_matrix(actual_labels_combined, predicted_labels_combined, labels=np.arange(len(label_list)))
    moving_indices = get_moving_indices(game)
    indices_to_keep = moving_indices + [label_list.index("not_an_object"), label_list.index("not_detected")]
    modified_label_list = [label for i, label in enumerate(label_list) if i in indices_to_keep]
    # remove rows and columns that are not relevant
    cm = cm[np.ix_(indices_to_keep, indices_to_keep)]
    plot_confusion_matrix(cm,
                          modified_label_list,
                          path= f"confusion_matrix_detection_{game}.png",
                          title = f"Confusion Matrix for {game.capitalize()}")
    return cm

def remove_not_detected(actual_labels_combined, predicted_labels_combined, not_detected_int):
    actual_labels_combined = np.array(actual_labels_combined)
    predicted_labels_combined = np.array(predicted_labels_combined)
    not_detected_indices = np.where(predicted_labels_combined == not_detected_int)[0]
    actual_labels_combined = np.delete(actual_labels_combined, not_detected_indices)
    predicted_labels_combined = np.delete(predicted_labels_combined, not_detected_indices)
    return actual_labels_combined, predicted_labels_combined

def remove_not_an_object(actual_labels_combined, predicted_labels_combined, not_an_object_int):
    actual_labels_combined = np.array(actual_labels_combined)
    predicted_labels_combined = np.array(predicted_labels_combined)
    not_an_object_indices = np.where(actual_labels_combined == not_an_object_int)[0]
    actual_labels_combined = np.delete(actual_labels_combined, not_an_object_indices)
    predicted_labels_combined = np.delete(predicted_labels_combined, not_an_object_indices)
    return actual_labels_combined, predicted_labels_combined

def compute_precision(actual_labels_combined, predicted_labels_combined, label_list: List[Union[str, int]]):
    actual_labels_combined, predicted_labels_combined = remove_not_detected(actual_labels_combined, predicted_labels_combined, label_list.index("not_detected"))
    cm = confusion_matrix(actual_labels_combined, predicted_labels_combined, labels=np.arange(len(label_list)))
    precision = np.sum(np.diag(cm))/ np.sum(cm)
    return precision

def compute_recall(actual_labels_combined, predicted_labels_combined, label_list: List[Union[str, int]]):
    actual_labels_combined, predicted_labels_combined = remove_not_an_object(actual_labels_combined, predicted_labels_combined, label_list.index("not_an_object"))
    cm = confusion_matrix(actual_labels_combined, predicted_labels_combined, labels=np.arange(len(label_list)))
    recall = np.sum(np.diag(cm))/ np.sum(cm)
    return recall

def compute_f1_score(recall, precision):
    return 2 * (recall * precision) / (recall + precision)
