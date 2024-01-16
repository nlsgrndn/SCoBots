from sklearn.metrics import confusion_matrix
from dataset.atari_labels import label_list_for,  get_moving_indices
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union
from eval.utils import plot_confusion_matrix
from utils.bbox_matching import match_bounding_boxes
from dataset.z_what import Atari_Z_What
from engine.utils import get_config_v2
from torch.utils.data import DataLoader
from motrackers import load_classifier

def confusion_matrix_visualization(dataloader, classifier, centroid_labels_dict, label_list: List[Union[str, int]]):
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
            pred_labels = np.array([centroid_labels_dict[label] for label in pred_labels])
            curr_pred_bbs_and_labels = np.concatenate((curr_pred_boxes, pred_labels[:, np.newaxis]), axis=1)
            actual_labels, predicted_labels = match_bounding_boxes(curr_gt_bbs_and_labels, curr_pred_bbs_and_labels, label_list) # len(actual) = len(labels[i]) + # of predicted boxes that are not matched
            actual_labels_combined.extend(actual_labels)
            predicted_labels_combined.extend(predicted_labels)

    label_list.append("not_an_object") # add no_object for case that no bounding box is predicted or predicted bounding box is not matched
    label_list.append("not_detected") # add not_detected for case that no bounding box is predicted or predicted bounding box is not matched
    cm = confusion_matrix(actual_labels_combined, predicted_labels_combined, labels=np.arange(len(label_list)))
    plot_confusion_matrix(cm, label_list, path= "confusion_matrix_detection_and_classification.png")
    return cm

if __name__ == "__main__":
    #FILTERED_PREDICTED_BBS = True
    game = "skiing"
    data_subset_mode = "relevant"
    #labels = np.load(f"labeled/{game}/actual_bbs_test.npz")
    #predicted = np.load(f"labeled/{game}/predbboxs_predlabels_gtlabels_z_whats_test_{'filtered' if FILTERED_PREDICTED_BBS else 'unfiltered'}.npz")
    cfg = get_config_v2(f"configs/my_atari_{game}_gpu.yaml")
    dataset = Atari_Z_What(cfg, "test", boxes_subset=data_subset_mode, return_keys=["gt_bbs_and_labels", "pred_boxes", "z_whats_pres_s"])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    clf, centroid_labels_dict= load_classifier(game)
    label_list = label_list_for(game)
    cm = confusion_matrix_visualization(dataloader, clf, centroid_labels_dict, label_list)
    print(cm)