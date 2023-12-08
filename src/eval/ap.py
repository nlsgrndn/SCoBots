import numpy as np


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

def compute_ap(pred_boxes, gt_boxes, iou_thresholds=None, recall_values=None):
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
        # Number of total ground truths
        count_gt = 0
        hit = []
        # For each image, determine each prediction is a hit of not
        for pred, gt in zip(pred_boxes, gt_boxes):
            count_gt += len(gt)
            # Sort predictions within an image by decreasing confidence
            pred = sorted(pred, key=lambda x: -x[-1])

            if len(gt) == 0:
                hit.extend((False, conf) for *_, conf in pred)
                continue
            if len(pred) == 0:
                continue

            M, N = len(pred), len(gt)

            # (M, 4), (M) (N, 4)
            pred = np.array(pred)
            pred, conf = pred[:, :4], pred[:, -1]
            gt = np.array(gt)

            # (M, N)
            iou = compute_iou(pred, gt)
            # (M,)
            best_indices = np.argmax(iou, axis=1)
            # (M,)
            best_iou = iou[np.arange(M), best_indices]
            # (N,), thresholding results
            valid = best_iou > threshold
            used = [False] * N

            for i in range(M):
                # Only count first hit
                if valid[i] and not used[best_indices[i]]:
                    hit.append((True, conf[i]))
                    used[best_indices[i]] = True
                else:
                    hit.append((False, conf[i]))

        if len(hit) == 0:
            AP.append(0.0)
            continue

        # Sort
        # hit.sort(key=lambda x: -x[-1])
        hit = sorted(hit, key=lambda x: -x[-1])

        # print('yes')
        # print(hit)
        hit = [x[0] for x in hit]
        # print(hit)
        # Compute average precision
        # hit_cum = np.cumsum(np.array(hit, dtype=np.float32))
        hit_cum = np.cumsum(hit)
        num_cum = np.arange(len(hit)) + 1.0
        precision = hit_cum / num_cum
        recall = hit_cum / count_gt
        # Compute AP at selected recall values
        # print(precision)
        precs = []
        for val in recall_values:
            prec = precision[recall >= val]
            precs.append(0.0 if prec.size == 0 else prec.max())

        # Mean over recall values
        AP.append(np.mean(precs))
        # AP.extend(precs)

        # print(AP)
    print(AP)
    # mean over all thresholds
    return AP


def compute_prec_rec(pred_boxes, gt_boxes, threshold_values):
    """
    Compute precision and recall using misalignment.

    :param pred_boxes: [[y_min, y_max, x_min, x_max, conf] * N] * B
    :param gt_boxes: [[y_min, y_max, x_min, x_max] * N] * B
    :return: p/r
    """
    threshold = 0.5
    count_gt = 0
    hit = []
    # For each image, determine each prediction is a hit of not
    for pred, gt in zip(pred_boxes, gt_boxes):
        count_gt += len(gt)
        # Sort predictions within an image by decreasing confidence
        pred = sorted(pred, key=lambda x: -x[-1])

        if len(gt) == 0:
            hit.extend((False, conf) for *_, conf in pred)
            continue
        if len(pred) == 0:
            continue

        M, N = len(pred), len(gt)

        # (M, 4), (M) (N, 4)
        pred = np.array(pred)
        pred, conf = pred[:, :4], pred[:, -1]
        gt = np.array(gt)

        # (M, N)
        mis_align = compute_misalignment(pred, gt)
        # (M,)
        best_indices = np.argmax(mis_align, axis=1)
        # (M,)
        best_mis_align = mis_align[np.arange(M), best_indices]
        # (N,), thresholding results
        valid = best_mis_align > threshold
        used = [False] * N

        for i in range(M):
            # Only count first hit
            if valid[i] and not used[best_indices[i]]:
                hit.append((True, conf[i]))
                used[best_indices[i]] = True
            else:
                hit.append((False, conf[i]))

    hit = sorted(hit, key=lambda x: -x[-1]) # sort by confidence (descending)
    confidence_values = np.array([x[1] for x in hit])
    hit = [x[0] for x in hit] # remove confidence
    hit_cum = np.cumsum(hit)
    num_cum = np.arange(len(hit)) + 1.0
    precision = hit_cum / num_cum
    recall = hit_cum / count_gt
    if len(precision) == 0 or len(recall) == 0:
        return 0.0, 0.0, np.array([0.0]), np.array([0.0]), np.array([0.0])
    
    precisions_at_thresholds = np.zeros(len(threshold_values))
    recalls_at_thresholds = np.zeros(len(threshold_values))
    # store precision and recall for different confidence thresholds
    for i, threshold in enumerate(threshold_values):
        # get first index where confidence is above threshold
        index = np.argmin(confidence_values[confidence_values > threshold]) if confidence_values[confidence_values > threshold].size > 0 else None
        if index is None:
            precisions_at_thresholds[i] = np.nan
            recalls_at_thresholds[i] = np.nan
        else:
            precisions_at_thresholds[i] = precision[index]
            recalls_at_thresholds[i] = recall[index]
    return precision[-1], recall[-1], precisions_at_thresholds, recalls_at_thresholds


def compute_iou(pred, gt):
    """

    :param pred: (M, 4), (y_min, y_max, x_min, x_max)
    :param gt: (N, 4)
    :return: (M, N)
    """
    compute_area = lambda b: (b[:, 1] - b[:, 0]) * (b[:, 3] - b[:, 2])

    area_pred = compute_area(pred)[:, None]
    area_gt = compute_area(gt)[None, :]
    # (M, 1, 4), (1, N, 4)
    pred = pred[:, None]
    gt = gt[None, :]
    # (M, 1) (1, N)

    # (M, N)
    top = np.maximum(pred[:, :, 0], gt[:, :, 0])
    bottom = np.minimum(pred[:, :, 1], gt[:, :, 1])
    left = np.maximum(pred[:, :, 2], gt[:, :, 2])
    right = np.minimum(pred[:, :, 3], gt[:, :, 3])

    h_inter = np.maximum(0.0, bottom - top)
    w_inter = np.maximum(0.0, right - left)
    # (M, N)
    area_inter = h_inter * w_inter
    iou = area_inter / (area_pred + area_gt - area_inter)

    return iou


def compute_diagonal(bbs):
    return np.sqrt((bbs[:, 1] - bbs[:, 0]) ** 2 + (bbs[:, 3] - bbs[:, 2]) ** 2)


def compute_misalignment(pred, gt):
    diagonal_gt = compute_diagonal(gt)
    center_pred = np.column_stack([(pred[:, 0] + pred[:, 1]) / 2, (pred[:, 2] + pred[:, 3]) / 2])[:, None, :]
    center_gt = np.column_stack([(gt[:, 0] + gt[:, 1]) / 2, (gt[:, 2] + gt[:, 3]) / 2])[None, :, :]
    delta_bbs = np.concatenate([
        np.repeat(center_pred, len(gt), axis=1),
        np.repeat(center_gt, len(pred), axis=0)
    ], axis=2)
    delta_bbs[:, :, [1, 2]] = delta_bbs[:, :, [2, 1]]
    diagonal_pair = compute_diagonal(delta_bbs.reshape((-1, 4))).reshape((len(pred), len(gt)))
    return np.maximum(0, 1 - diagonal_pair / diagonal_gt)
