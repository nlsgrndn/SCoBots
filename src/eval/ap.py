import numpy as np
from utils.bbox_matching import compute_iou, compute_misalignment, compute_hits, match_bounding_boxes_center_distances

def compute_counts(boxes_pred, boxes_gt):
    """
    Compute error rates, perfect number, overcount number, undercount number

    :param boxes_pred: [[y_min, y_max, x_min, x_max, conf] * N] * B
    :param boxes_gt: [[y_min, y_max, x_min, x_max] * N] * B
    :return:
        error_rate: mean of ratio of differences
        perfect: integer
        overcount: integer
        undercount: integer
    """

    error_rates = []
    perfect = 0
    overcount = 0
    undercount = 0
    for pred, gt in zip(boxes_pred, boxes_gt):
        M, N = len(pred), len(gt)
        error_rates.append(abs(M - N) / N)
        perfect += (M == N)
        overcount += (M > N)
        undercount += (M < N)

    return np.mean(error_rates), perfect, overcount, undercount

def compute_ap(pred_boxes, gt_boxes, iou_thresholds=None, recall_values=None, matching_method=compute_iou):
    """
    Compute average precision over different iou thresholds.

    :param pred_boxes: [[y_min, y_max, x_min, x_max, conf] * N] * B
    :param gt_boxes: [[y_min, y_max, x_min, x_max] * N] * B
    :param iou_thresholds: a list of iou thresholds
    :param recall_values: a list of recall values to compute AP
    :return: AP at each iou threshold
    """
    if recall_values is None:
        recall_values = np.linspace(0.0, 1.0, 11)

    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.1, 0.9, 9)

    AP = []
    for threshold in iou_thresholds:
        # Compute AP for this threshold
        
        hit, count_gt = compute_hits(pred_boxes, gt_boxes, threshold, matching_method)

        if len(hit) == 0:
            AP.append(0.0)
            continue

        # Sort
        hit = sorted(hit, key=lambda x: -x[-1])
        hit = [x[0] for x in hit]
        
        # Compute average precision
        hit_cum = np.cumsum(hit)
        num_cum = np.arange(len(hit)) + 1.0
        precision = hit_cum / num_cum
        recall = hit_cum / count_gt
        
        # Compute AP at selected recall values
        precs = []
        for val in recall_values:
            prec = precision[recall >= val]
            precs.append(0.0 if prec.size == 0 else prec.max())

        # Mean over recall values
        AP.append(np.mean(precs))

    print(AP)
    return AP


def compute_prec_rec(pred_boxes, gt_boxes, threshold_values, matching_method=compute_misalignment):
    """
    Compute precision and recall using misalignment.

    :param pred_boxes: [[y_min, y_max, x_min, x_max, conf] * N] * B
    :param gt_boxes: [[y_min, y_max, x_min, x_max] * N] * B
    :return: p/r
    """
    threshold = 0.5
    hit, count_gt = compute_hits(pred_boxes, gt_boxes, threshold, matching_method)

    hit = sorted(hit, key=lambda x: -x[-1]) # sort by confidence (descending)
    confidence_values = np.array([x[1] for x in hit])
    hit = [x[0] for x in hit] # remove confidence
    hit_cum = np.cumsum(hit)
    num_cum = np.arange(len(hit)) + 1.0
    precision = hit_cum / num_cum
    recall = hit_cum / count_gt
    if len(precision) == 0 or len(recall) == 0:
        return 0.0, 0.0, np.zeros(len(threshold_values)), np.zeros(len(threshold_values))
    
    precisions_at_thresholds = np.zeros(len(threshold_values))
    recalls_at_thresholds = np.zeros(len(threshold_values))
    # store precision and recall for different confidence thresholds
    for i, thresh in enumerate(threshold_values):
        # get first index where confidence is above threshold
        index = np.argmin(confidence_values[confidence_values > thresh]) if confidence_values[confidence_values > thresh].size > 0 else None
        if index is None:
            precisions_at_thresholds[i] = np.nan
            recalls_at_thresholds[i] = np.nan
        else:
            precisions_at_thresholds[i] = precision[index]
            recalls_at_thresholds[i] = recall[index]
    return precision[-1], recall[-1], precisions_at_thresholds, recalls_at_thresholds


def compute_average_center_distances(pred_boxes, gt_boxes):
    """
    :param pred_boxes: [[y_min, y_max, x_min, x_max, conf] * N] * B
    :param gt_boxes: [[y_min, y_max, x_min, x_max] * N] * B
    :return: average center distance
    """

    distances = []
    for pred, gt in zip(pred_boxes, gt_boxes):
        curr_distances = match_bounding_boxes_center_distances(gt, pred)
        distances.extend(curr_distances)
    return np.mean(distances)


