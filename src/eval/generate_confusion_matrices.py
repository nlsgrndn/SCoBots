from sklearn.metrics import confusion_matrix
from dataset.atari_labels import label_list_for
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union
from eval.utils import plot_confusion_matrix
from utils.bbox_matching import match_bounding_boxes

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

if __name__ == "__main__":
    FILTERED_PREDICTED_BBS = True
    game = "boxing"
    labels = np.load(f"labeled/{game}/actual_bbs_test.npz")
    predicted = np.load(f"labeled/{game}/predbboxs_predlabels_gtlabels_z_whats_test_{'filtered' if FILTERED_PREDICTED_BBS else 'unfiltered'}.npz")
    print(confusion_matrix_visualization(list(labels.values()), list(predicted.values()), label_list_for(game)))