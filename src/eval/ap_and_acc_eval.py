import numpy as np
import torch

from model.space.postprocess_latent_variables import convert_to_boxes, retrieve_latent_repr_from_logs
from .eval_cfg import eval_cfg
from .ap import read_boxes, compute_ap, compute_counts, compute_prec_rec, read_boxes_object_type_dict
from dataset import get_label_list

class ApAndAccEval():
    @torch.no_grad()
    def eval_ap_and_acc(self, logs, dataset, bb_path, iou_thresholds=None):
        """
        Evaluate average precision and accuracy
        :param logs: the model output
        :param dataset: the dataset for accessing label information
        :param bb_path: directory containing the gt bounding boxes.
        :param iou_thresholds:
        :return ap: a list of average precisions, corresponding to each iou_thresholds
        """
        batch_size = eval_cfg.train.batch_size
        num_samples = min(len(dataset), eval_cfg.train.num_samples.ap)
        print('Computing error rates, counts and APs...')
        if iou_thresholds is None:
            iou_thresholds = np.linspace(0.1, 0.9, 9)
        boxes_gt_types = ['all', 'moving', 'relevant']
        indices = list(range(num_samples))
        boxes_gts = {k: v for k, v in zip(boxes_gt_types, read_boxes(bb_path, indices=indices))} # boxes_gts['moving'] and boxes_gts['relevant'] are actually equivalent
        boxes_pred = []
        boxes_relevant = []

        num_batches = min(len(dataset), eval_cfg.train.num_samples.cluster) // batch_size #eval_cfg.train.num_samples.cluster // eval_cfg.train.batch_size

        # generating predicted bounding boxes from latent variables
        for img in logs[:num_batches]:
            z_where, z_pres, z_pres_prob, _ = retrieve_latent_repr_from_logs(img)
            boxes_batch = convert_to_boxes(z_where, z_pres, z_pres_prob, with_conf=True)
            boxes_relevant.extend(dataset.filter_relevant_boxes(boxes_batch, boxes_gts['all'])) # uses handcrafted rules to filter out irrelevant boxes for every game; boxes_gt['all'] is only used for one game
            boxes_pred.extend(boxes_batch)

        result = {}
        # Comparing predicted bounding boxes with ground truth
        for gt_name, gt in boxes_gts.items():
            # Four numbers
            boxes = boxes_pred if gt_name != "relevant" else boxes_relevant
            error_rate, perfect, overcount, undercount = compute_counts(boxes, gt)
            accuracy = perfect / (perfect + overcount + undercount)
            result[f'error_rate_{gt_name}'] = error_rate
            result[f'perfect_{gt_name}'] = perfect
            result[f'accuracy_{gt_name}'] = accuracy
            result[f'overcount_{gt_name}'] = overcount
            result[f'undercount_{gt_name}'] = undercount
            result[f'iou_thresholds_{gt_name}'] = iou_thresholds
            # A list of length 9 and P/R from low IOU level = 0.2
            aps = compute_ap(boxes, gt, iou_thresholds)
            precision, recall, precisions, recalls, thresholds = compute_prec_rec(boxes, gt)
            result[f'APs_{gt_name}'] = aps
            result[f'precision_{gt_name}'] = precision
            result[f'recall_{gt_name}'] = recall
            result[f'precisions_{gt_name}'] = precisions
            result[f'recalls_{gt_name}'] = recalls
            result[f'thresholds_{gt_name}'] = thresholds

        # compute recall for object types
        #boxes_of_label = read_boxes_object_type_dict(bb_path, None, indices=indices)
        #for label in boxes_of_label.keys():
        #    _, recall = compute_prec_rec(boxes_pred, boxes_of_label[label])
        #    result[f'recall_{label}'] = recall
        
        return result