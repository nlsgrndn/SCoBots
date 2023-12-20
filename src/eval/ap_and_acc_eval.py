import numpy as np
import torch

from model.space.postprocess_latent_variables import convert_to_boxes, retrieve_latent_repr_from_logs
from .eval_cfg import eval_cfg
from .ap import compute_ap, compute_counts, compute_prec_rec
from eval.data_reading import read_boxes, read_boxes_object_type_dict
from dataset.z_what import Atari_Z_What
from torch.utils.data import DataLoader

class ApAndAccEval():
    AP_IOU_THRESHOLDS = np.linspace(0.1, 0.9, 9)
    PREC_REC_CONF_THRESHOLDS = np.append(np.arange(0.5, 0.95, 0.05), np.arange(0.95, 1.0, 0.01))

    @torch.no_grad()
    def eval_ap_and_acc(self, logs, dataset, bb_path, data_subset_modes, cfg):
        """
        Evaluate average precision and accuracy
        :param logs: the model output
        :param dataset: the dataset for accessing label information
        :param bb_path: directory containing the gt bounding boxes.
        :return ap: a list of average precisions, corresponding to each iou_thresholds
        """
        print('Computing error rates, counts and APs...')
        data = self.get_bbox_data_via_dataloader(cfg, data_subset_modes) #self.get_bbox_data(logs, dataset, bb_path)#TODO remove comment
        results = self.compute_metrics(data)
        return results
    
    #def get_bbox_data(self, logs, dataset, bb_path,):
    #    # read ground truth bounding boxes
#
    #    num_samples = min(len(dataset), eval_cfg.train.num_samples.ap)
    #    num_batches = num_samples // eval_cfg.train.batch_size
#
    #    indices = list(range(num_samples))
    #    boxes_gt_types = ['all', 'moving', 'relevant']
    #    boxes_gts = {k: v for k, v in zip(boxes_gt_types, read_boxes(bb_path, indices=indices))} # boxes_gts['moving'] and boxes_gts['relevant'] are actually equivalent
    #    
    #    # collect and generate predicted bounding boxes
    #    boxes_pred = []
    #    boxes_pred_relevant = []
    #    for img in logs[:num_batches]:
    #        z_where, z_pres, z_pres_prob, _ = retrieve_latent_repr_from_logs(img)
    #        boxes_batch = convert_to_boxes(z_where, z_pres, z_pres_prob, with_conf=True)
    #        boxes_pred.extend(boxes_batch)
    #        boxes_pred_relevant.extend(dataset.filter_relevant_boxes(boxes_batch, boxes_gts['all'])) # uses handcrafted rules to filter out irrelevant boxes for every game; boxes_gt['all'] is only used for one game
#
    #    data = {"all": (boxes_gts['all'], boxes_pred),
    #            "moving": (boxes_gts['moving'], boxes_pred),
    #            "relevant": (boxes_gts['relevant'], boxes_pred_relevant)}
    #    
    #    import ipdb; ipdb.set_trace()
    #    return data
    
    def get_bbox_data_via_dataloader(self, cfg, data_subset_modes):
        dataset_mode = "test" #TODO: pass as argument or get from cfg in some way
        data = {} 
        for boxes_subset in data_subset_modes:
            atari_z_what_dataset = Atari_Z_What(cfg, dataset_mode, boxes_subset, return_keys = ["pred_boxes", "gt_bbs_and_labels"])
            atari_z_what_dataloader = DataLoader(atari_z_what_dataset, batch_size=1, shuffle=False, num_workers=0) #batch_size must be 1
            boxes_pred, boxes_gt = [], []
            for batch in atari_z_what_dataloader:
                boxes_pred.extend([np.array(batch["pred_boxes"][i][0]) for i in range(len(batch["pred_boxes"]))]) #[0] because batch_size = 1 and dataloader implicitly adds a batch dimension
                boxes_gt.extend([np.array(batch["gt_bbs_and_labels"][i][0]) for i in range(len(batch["gt_bbs_and_labels"]))]) #[0] because batch_size = 1 and dataloader implicitly adds a batch dimension
            data[boxes_subset] = (boxes_gt, boxes_pred)
        return data

    def compute_metrics(self, data):
        result = {}
        # Comparing predicted bounding boxes with ground truth
        for gt_name, (gt_boxes, pred_boxes) in data.items():
            # compute results
            error_rate, perfect, overcount, undercount = compute_counts(pred_boxes, gt_boxes)
            accuracy = perfect / (perfect + overcount + undercount)
            # A list of length 9 and P/R from low IOU level = 0.2s
            aps = compute_ap(pred_boxes, gt_boxes, self.AP_IOU_THRESHOLDS)
            precision, recall, precisions, recalls = compute_prec_rec(pred_boxes, gt_boxes, self.PREC_REC_CONF_THRESHOLDS)

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

        # compute recall for object types
        #boxes_of_label = read_boxes_object_type_dict(bb_path, None, indices=indices)
        #for label in boxes_of_label.keys():
        #    _, recall = compute_prec_rec(boxes_pred, boxes_of_label[label])
        #    result[f'recall_{label}'] = recall
        
        return result
