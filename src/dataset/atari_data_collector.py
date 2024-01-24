import torch
from dataset.z_what import Atari_Z_What
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np

class AtariDataCollector:

    return_keys_with_list_return_value = ["z_whats_pres_s", "gt_labels_for_pred_boxes", "pred_boxes", "gt_bbs_and_labels"]

    def __init__(self):
        pass
    
    @staticmethod
    def collect_atari_data(cfg, dataset_mode, data_subset_mode, return_keys, only_collect_first_image_of_consecutive_frames):    
        atari_z_what_dataset = Atari_Z_What(cfg, dataset_mode, data_subset_mode, return_keys = return_keys)
        atari_z_what_dataloader = DataLoader(atari_z_what_dataset, batch_size=1, shuffle=False, num_workers=0) #batch_size must be 1
        result_dict = defaultdict(list)
        T = atari_z_what_dataset.T
        if only_collect_first_image_of_consecutive_frames:
            T = 1
        for batch in atari_z_what_dataloader:
            for return_key in return_keys:
                if return_key in AtariDataCollector.return_keys_with_list_return_value:
                    value_for_current_return_key = batch[return_key]
                    tmp_value = [value_for_current_return_key[i][0] for i in range(T)] #[0] because batch_size = 1 and dataloader implicitly adds a batch dimension
                    result_dict[return_key].extend(tmp_value)
                else:
                    result_dict[return_key].extend(batch[return_key][0][0:T]) #[0] because batch_size = 1 and dataloader implicitly adds a batch dimension
        return result_dict
    
    #convencience methods
    @staticmethod
    def collect_z_what_data_reshaped(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames=False):        
        result_dict = AtariDataCollector.collect_atari_data(cfg, dataset_mode, data_subset_mode, return_keys=["z_whats_pres_s", "gt_labels_for_pred_boxes"], only_collect_first_image_of_consecutive_frames=only_collect_first_image_of_consecutive_frames)
        z_whats = torch.cat(result_dict["z_whats_pres_s"], dim=0)
        labels= torch.cat(result_dict["gt_labels_for_pred_boxes"], dim=0)
        labels = labels.squeeze(-1)
        return z_whats, labels
    
    @staticmethod
    def collect_z_what_data(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames=False):
        result_dict = AtariDataCollector.collect_atari_data(cfg, dataset_mode, data_subset_mode, return_keys=["z_whats_pres_s"], only_collect_first_image_of_consecutive_frames=only_collect_first_image_of_consecutive_frames)
        return result_dict["z_whats_pres_s"], result_dict["gt_labels_for_pred_boxes"]
    
    @staticmethod
    def collect_images(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames=False):
        result_dict = AtariDataCollector.collect_atari_data(cfg, dataset_mode, data_subset_mode, return_keys=["imgs"], only_collect_first_image_of_consecutive_frames=only_collect_first_image_of_consecutive_frames)
        images = torch.stack(result_dict["imgs"], dim=0)
        return images
    
    @staticmethod
    def collect_pred_boxes(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames=False):
        result_dict = AtariDataCollector.collect_atari_data(cfg, dataset_mode, data_subset_mode, return_keys=["pred_boxes"], only_collect_first_image_of_consecutive_frames=only_collect_first_image_of_consecutive_frames)
        pred_boxes = result_dict["pred_boxes"]
        return pred_boxes
    
    @staticmethod
    def collect_bbox_data(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames=False):
        result_dict = AtariDataCollector.collect_atari_data(cfg, dataset_mode, data_subset_mode, return_keys=["pred_boxes", "gt_bbs_and_labels"], only_collect_first_image_of_consecutive_frames=only_collect_first_image_of_consecutive_frames)
        pred_boxes = [np.array(pred_boxes) for pred_boxes in result_dict["pred_boxes"]]
        gt_boxes = [np.array(gt_boxes) for gt_boxes in result_dict["gt_bbs_and_labels"]]
        return gt_boxes, pred_boxes
    
                    

