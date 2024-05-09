import numpy as np
import torch

from  space_models.space.postprocess_latent_variables import convert_to_boxes, retrieve_latent_repr_from_logs
from .ap import compute_ap, compute_counts, compute_prec_rec, compute_average_center_distances
from eval.data_reading import read_boxes, read_boxes_object_type_dict
from dataset.z_what import Atari_Z_What
from dataset.atari_data_collector import AtariDataCollector
from torch.utils.data import DataLoader
from dataset.atari_labels import get_moving_indices, label_list_for

class ApAndAccEval():
    AP_IOU_THRESHOLDS = np.linspace(0.1, 0.9, 9)
    PREC_REC_CONF_THRESHOLDS = np.append(np.arange(0.5, 0.95, 0.05), np.arange(0.95, 1.0, 0.01))

    def __init__(self, cfg):
        self.cfg = cfg
        self.game = cfg.gamelist[0]

    @torch.no_grad()
    def eval_ap_and_acc(self, data_subset_modes, dataset_mode):
        """
        Evaluate average precision and accuracy
        :param logs: the model output
        :param dataset: the dataset for accessing label information
        :param bb_path: directory containing the gt bounding boxes.
        :return ap: a list of average precisions, corresponding to each iou_thresholds
        """
        print('Computing error rates, counts and APs...')
        data = self.get_bbox_data_via_dataloader(data_subset_modes, dataset_mode)
        results = self.compute_metrics(data)
        return results
    
    def get_bbox_data_via_dataloader(self, data_subset_modes, dataset_mode):
        data = {}
        for boxes_subset in data_subset_modes:
            data[boxes_subset] = AtariDataCollector.collect_bbox_data(self.cfg, dataset_mode, boxes_subset, only_collect_first_image_of_consecutive_frames=False)
        return data

    def compute_metrics(self, data):
        result = {}
        complete_label_list = label_list_for(self.game)
        labels = {"all": np.arange(len(complete_label_list)), "relevant": get_moving_indices(self.game)}
        # Comparing predicted bounding boxes with ground truth
        for gt_name, (gt_boxes, pred_boxes) in data.items():
            # compute results
            error_rate, perfect, overcount, undercount = compute_counts(pred_boxes, gt_boxes)
            accuracy = perfect / (perfect + overcount + undercount)
            # A list of length 9 and P/R from low IOU level = 0.2s
            aps = compute_ap(pred_boxes, gt_boxes, self.AP_IOU_THRESHOLDS)
            precision, recall, precisions, recalls = compute_prec_rec(pred_boxes, gt_boxes, self.PREC_REC_CONF_THRESHOLDS)
            average_distance = compute_average_center_distances(pred_boxes, gt_boxes)

            # store results
            result[f'error_rate_{gt_name}'] = error_rate
            result[f'perfect_{gt_name}'] = perfect
            result[f'overcount_{gt_name}'] = overcount
            result[f'undercount_{gt_name}'] = undercount
            result[f'accuracy_{gt_name}'] = accuracy
            result[f'APs_{gt_name}'] = aps
            result[f'precision_{gt_name}'] = precision
            result[f'recall_{gt_name}'] = recall
            result[f'precisions_{gt_name}'] = precisions
            result[f'recalls_{gt_name}'] = recalls
            result[f'average_distance_{gt_name}'] = average_distance

            # compute recall for object types
            for label in labels[gt_name]:
                # filter boxes by label
                gt_boxes_of_label = [gt_box_arr[gt_box_arr[:, 5] == label] for gt_box_arr in gt_boxes]
                precision, recall, precisions, recalls = compute_prec_rec(pred_boxes, gt_boxes_of_label, self.PREC_REC_CONF_THRESHOLDS)
                result[f'recall_label_{label}_{gt_name}'] = recall
        return result
