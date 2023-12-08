# TODO move to eval folder and integrate somehow with eval.py
from sklearn.metrics import confusion_matrix
from eval.ap import compute_iou    
from dataset.labels import label_list_for
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union
from eval.utils import plot_confusion_matrix

def confusion_matrix_visualization(labels: List[np.ndarray], predicted: List[np.ndarray], label_list: List[Union[str, int]]):
    actual_labels_combined = []
    predicted_labels_combined = []
    for i in range(len(labels)):
        actual_labels, predicted_labels = match_bounding_boxes(labels[i], predicted[i], label_list) # len(actual) = len(labels[i]) + # of predicted boxes that are not matched
        actual_labels_combined.extend(actual_labels)
        predicted_labels_combined.extend(predicted_labels)
    label_list.append("no_object") # add no_object for case that no bounding box is predicted or predicted bounding box is not matched
    cm = confusion_matrix(actual_labels_combined, predicted_labels_combined, labels=np.arange(len(label_list)))
    plot_confusion_matrix(cm, label_list, path= "confusion_matrix_detection_and_classification.png")
    return cm

def match_bounding_boxes(labels: np.ndarray, predicted: np.ndarray, label_list: List[Union[str, int]]):
    """
    Match bounding boxes in labels and predicted.
    :param labels: np.ndarray of shape (n, 5) where n is the number of bounding boxes
    :param predicted: np.ndarray of shape (m, 5) where m is the number of bounding boxes
    :return: actual_list, predicted_list
    """
    actual_list = []
    predicted_list = []
    label_idx_used = []
    NO_OBJECT = len(label_list)
    IOU_THRESHOLD = 0.5
    # compute iou between each pair of bounding boxes
    ious = compute_iou(predicted, labels)

    # match bounding boxes
    for i in range(predicted.shape[0]):
        max_iou = np.max(ious[i])
        if max_iou > IOU_THRESHOLD:
            actual_list.append(labels[np.argmax(ious[i])][4])
            predicted_list.append(predicted[i][4])
            label_idx_used.append(np.argmax(ious[i]))
        else:
            predicted_list.append(predicted[i][4])
            actual_list.append(NO_OBJECT)

    # add unmatched bounding boxes in labels
    for i in range(labels.shape[0]):
        if i not in label_idx_used:
            actual_list.append(labels[i][4])
            predicted_list.append(NO_OBJECT)

    return actual_list, predicted_list

if __name__ == "__main__":
    game = "pong"
    labels = np.load(f"labeled/{game}/actual_bbs_test.npz")
    predicted = np.load(f"labeled/{game}/predicted_bbs_test.npz")
    print(confusion_matrix_visualization(list(labels.values()), list(predicted.values()), label_list_for(game)))