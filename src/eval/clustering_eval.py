import torch
from .kalman_filter import classify_encodings
from .classify_z_what import ZWhatEvaluator
from eval.create_z_what_dataset import ZWhatDataCollector
import numpy as np

class ClusteringEval:

    def __init__(self, cfg, relevant_object_hover_path,):
        self.cfg = cfg
        self.relevant_object_hover_path = relevant_object_hover_path
    @torch.no_grad()
    def eval_clustering(self, logs, dataset, global_step, data_subset_modes):
        """
        Evaluate clustering metrics

        :param logs: results from applying the model
        :param dataset: dataset
        :param global_step: gradient step number
        :return metrics: for all classes of evaluation many metrics describing the clustering,
            based on different ground truths
        """
        #data, z_encs_relevant, labels_relevant_unflattened = \
        #    ZWhatDataCollector(self.cfg, self.relevant_object_hover_path).collect_z_what_data(logs, dataset, global_step,)
        data = ZWhatDataCollector(self.cfg, self.relevant_object_hover_path).collect_z_what_data(logs, dataset, global_step, self.cfg)
        results = {}
        for data_subset_mode in data_subset_modes:
            relevant_labels, test_x, test_y, train_x, train_y = data[data_subset_mode]
            mutual_info_scores_dict, self.dim_red_path, few_shot_accuracy_dict = ZWhatEvaluator(self.cfg, title= data_subset_mode,).evaluate_z_what(train_x, train_y, test_x, test_y, relevant_labels)
            if data_subset_mode == 'relevant' and few_shot_accuracy_dict is not None: #TODO uncomment or remove
            #    # add bayes accuracy metric for relevant objects based on Kalman filter
            #    bayes_accuracy = classify_encodings(self.cfg, z_encs_relevant, labels_relevant_unflattened)
            #    few_shot_accuracy_dict['bayes_accuracy'] = bayes_accuracy
                 few_shot_accuracy_dict['bayes_accuracy'] = np.nan
            results[data_subset_mode] = (mutual_info_scores_dict, self.dim_red_path, few_shot_accuracy_dict)
        return results



    