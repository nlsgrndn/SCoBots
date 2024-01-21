import torch
from .z_what_eval import ZWhatEvaluator
from eval.create_z_what_dataset import ZWhatDataCollector
import numpy as np

class ClusteringEval:

    def __init__(self, cfg):
        self.cfg = cfg

    @torch.no_grad()
    def eval_clustering(self, data_subset_modes, dataset_mode):
        """
        Evaluate clustering metrics

        :param logs: results from applying the model
        :param dataset: dataset
        :param global_step: gradient step number
        :return metrics: for all classes of evaluation many metrics describing the clustering,
            based on different ground truths
        """
        
        data = ZWhatDataCollector(self.cfg).collect_z_what_data(self.cfg, dataset_mode, data_subset_modes)
        for key in data: #TODO improve this
            relevant_labels, test_x, test_y, train_x, train_y = data[key]
            if test_x is not None and test_y is not None and train_x is not None and train_y is not None:
                test_x, test_y, train_x, train_y = test_x.cpu(), test_y.cpu(), train_x.cpu(), train_y.cpu()
            data[key] = (relevant_labels, test_x, test_y, train_x, train_y)
        results = {}
        for data_subset_mode in data_subset_modes:
            relevant_labels, test_x, test_y, train_x, train_y = data[data_subset_mode]
            mutual_info_scores_dict, self.dim_red_path, few_shot_accuracy_dict = ZWhatEvaluator(self.cfg, title= data_subset_mode,).evaluate_z_what(train_x, train_y, test_x, test_y, relevant_labels)

            results[data_subset_mode] = (mutual_info_scores_dict, self.dim_red_path, few_shot_accuracy_dict)
        return results



    