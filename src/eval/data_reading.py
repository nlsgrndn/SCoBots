from collections import defaultdict
import numpy as np
import pandas as pd
import os

def read_boxes(path, size=None, indices=None):
    """
    Read bounding boxes and normalize to (0, 1)
    Note BB files structure was changed to left, top coordinates + width and height
    :param path: checkpointdir to bounding box root
    :param size: how many indices
    :param indices: relevant indices of the dataset
    :return: A list of list [[y_min, y_max, x_min, x_max] * N] * B
    """
    boxes_all = []
    boxes_moving_all = []

    for stack_idx in indices if (indices is not None) else range(size):
        for img_idx in range(4):
            boxes = []
            boxes_moving = []
            bbs = pd.read_csv(os.path.join(path, f"{stack_idx:05}_{img_idx}.csv"), header=None, usecols=[0, 1, 2, 3, 4])
            for y_min, y_max, x_min, x_max, moving in bbs.itertuples(index=False, name=None):
                boxes.append([y_min, y_max, x_min, x_max])
                if "M" in moving:
                    boxes_moving.append([y_min, y_max, x_min, x_max])
            boxes = np.array(boxes)
            boxes_moving = np.array(boxes_moving)
            boxes_all.append(boxes)
            boxes_moving_all.append(boxes_moving)
    return boxes_all, boxes_moving_all, boxes_moving_all


def read_boxes_with_labels(path, size=None, indices=None):
    """
    Read bounding boxes and normalize to (0, 1)
    Note BB files structure was changed to left, top coordinates + width and height
    :param path: checkpointdir to bounding box root
    :param size: how many indices
    :param indices: relevant indices of the dataset
    :return: A list of list of list [[[y_min, y_max, x_min, x_max, label] * N] * B]
    """
    boxes_all = []
    boxes_moving_all = []

    for stack_idx in indices if (indices is not None) else range(size):
        for img_idx in range(4):
            boxes = []
            boxes_moving = []
            bbs = pd.read_csv(os.path.join(path, f"{stack_idx:05}_{img_idx}.csv"), header=None, usecols=[0, 1, 2, 3, 4, 5])
            for y_min, y_max, x_min, x_max, moving, label in bbs.itertuples(index=False, name=None):
                boxes.append([y_min, y_max, x_min, x_max, label])
                if "M" in moving:
                    boxes_moving.append([y_min, y_max, x_min, x_max, label])
            boxes = np.array(boxes)
            boxes_moving = np.array(boxes_moving)
            boxes_all.append(boxes)
            boxes_moving_all.append(boxes_moving)
    return boxes_all, boxes_moving_all, boxes_moving_all


def read_boxes_object_type_dict(path, size=None, indices=None):
    boxes_all_dict = defaultdict(list)
    for stack_idx in indices if (indices is not None) else range(size):
        for img_idx in range(4):
            boxes_tmp_dict = defaultdict(list)
            bbs = pd.read_csv(os.path.join(path, f"{stack_idx:05}_{img_idx}.csv"), header=None, usecols=[0, 1, 2, 3, 4, 5])
            for y_min, y_max, x_min, x_max, moving, label in bbs.itertuples(index=False, name=None):
                boxes_tmp_dict[label].append([y_min, y_max, x_min, x_max])
            for label in boxes_tmp_dict:
                boxes_all_dict[label].append(np.array(boxes_tmp_dict[label]))
    # transform to numpy arrays
    return boxes_all_dict