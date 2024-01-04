from eval.clustering_eval import ClusteringEval
from eval.ap_and_acc_eval import ApAndAccEval
from utils.metric_logger import MetricLogger
import numpy as np
import torch
import os
from torch.utils.data import Subset, DataLoader
from .eval_cfg import eval_cfg
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os
import pprint
from create_latent_dataset import create_latent_dataset



class SpaceEval:

    ap_results_none_dict = {
            'all': ({'adjusted_mutual_info_score': np.nan, 'adjusted_rand_score': np.nan}, "dummy_path",
                    {'few_shot_accuracy_with_1': np.nan, 'few_shot_accuracy_with_4': np.nan, 'few_shot_accuracy_with_16': np.nan,
                     'few_shot_accuracy_with_64': np.nan, 'few_shot_accuracy_cluster_nn': np.nan}),
            'moving': ({'adjusted_mutual_info_score': np.nan, 'adjusted_rand_score': np.nan}, "dummy_path",
                       {'few_shot_accuracy_with_1': np.nan, 'few_shot_accuracy_with_4': np.nan,
                        'few_shot_accuracy_with_16': np.nan,
                        'few_shot_accuracy_with_64': np.nan, 'few_shot_accuracy_cluster_nn': np.nan}),
            'relevant': ({'adjusted_mutual_info_score': np.nan, 'adjusted_rand_score': np.nan}, "dummy_path",
                         {'few_shot_accuracy_with_1': np.nan, 'few_shot_accuracy_with_4': np.nan,
                          'few_shot_accuracy_with_16': np.nan,
                          'few_shot_accuracy_with_64': np.nan, 'few_shot_accuracy_cluster_nn': np.nan, 'bayes_accuracy': np.nan})
        }

    def __init__(self, cfg, tb_writer, eval_mode):
        self.eval_mode = eval_mode
        self.eval_file_path = f'{cfg.logdir}/{cfg.exp_name}/{eval_mode}_metrics.csv'
        self.first_eval = True
        self.data_subset_modes = ['all','relevant']
        self.tb_writer = tb_writer
        self.file_writer = EvalWriter(cfg, tb_writer, self.data_subset_modes, self.eval_mode, self.eval_file_path)
        self.cfg = cfg
    
    @torch.no_grad()
    # @profile
    def eval(self, model, global_step, cfg,):
        """
        Evaluation. This includes:
            - mse evaluated on dataset
            - ap and accuracy evaluated on dataset
            - cluster metrics evaluated on dataset
        :return:
        """
        if self.first_eval:
            self.first_eval = False
            if os.path.exists(self.eval_file_path):
                os.remove(self.eval_file_path)
            self.file_writer.write_header()

        create_latent_dataset(cfg, self.eval_mode, model=model)
        
        results = self.core_eval_code(global_step, cfg,)
        return results

    def core_eval_code(self, global_step, cfg):
        results_collector = {}
        with open(self.eval_file_path, "a") as file:
            self.file_writer.write_metric(None, global_step, global_step, use_writer=False)
            if 'cluster' in eval_cfg.get(self.eval_mode).metrics:
                results = self.eval_clustering(global_step, cfg, self.eval_mode)
                results_collector.update(results)
                if cfg.train.log:
                    pp = pprint.PrettyPrinter(depth=2)
                    for res in results:
                        print("Cluster Result:")
                        pp.pprint(results[res])
            #if 'mse' in eval_cfg.get(self.eval_mode).metrics: #TODO make work again, currently broken because logs was removed
            #    mse = self.eval_mse(logs, global_step)
            #    print("MSE result: ", mse)
            #    results_collector.update({'mse': mse})
            if 'ap' in eval_cfg.get(self.eval_mode).metrics:
                results = self.eval_ap_and_acc(global_step, self.eval_mode)
                results_collector.update(results)
                if cfg.train.log:
                    results = {k2: v2[len(v2) // 4] if isinstance(v2, list) or isinstance(v2, np.ndarray) else v2 for
                               k2, v2, in
                               results.items()}
                    pp = pprint.PrettyPrinter(depth=2)
                    print("AP Result:")
                    pp.pprint({k: v for k, v in results.items() if "iou" not in k})
            file.write("\n")
        return results_collector

    @torch.no_grad()
    def eval_ap_and_acc(self, global_step, eval_mode):
        """
        Evaluate ap and accuracy
        :return: result_dict
        """
        result_dict = ApAndAccEval().eval_ap_and_acc(self.data_subset_modes, self.cfg, eval_mode)
        for class_name in self.data_subset_modes:
            APs = result_dict[f'APs_{class_name}']
            
            # only logging
            for ap, thres in zip(APs[1::4], ApAndAccEval.AP_IOU_THRESHOLDS[1::4]):
                self.tb_writer.add_scalar(f'val_aps_{class_name}/ap_{thres:.1}', ap, global_step)
            self.tb_writer.add_scalar(f'{class_name}/ap_avg_0.5', APs[len(APs) // 2], global_step)
            self.tb_writer.add_scalar(f'{class_name}/ap_avg_up', np.mean(APs[len(APs) // 2:]), global_step)
            self.tb_writer.add_scalar(f'{class_name}/ap_avg', np.mean(APs), global_step)
            
            # writing to file (and potentially logging)
            for ap in APs:
                self.file_writer.write_metric('ignored', ap, global_step, use_writer=False)
            metrics = ['accuracy', 'perfect', 'overcount', 'undercount', 'error_rate', 'precision', 'recall']
            self.file_writer.write_metrics(metrics, class_name, result_dict, global_step)
            for i, threshold in enumerate(ApAndAccEval.PREC_REC_CONF_THRESHOLDS):
                self.file_writer.write_metric(f'{class_name}/precision_at_{threshold}', result_dict[f'precisions_{class_name}'][i], global_step)
            for i, threshold in enumerate(ApAndAccEval.PREC_REC_CONF_THRESHOLDS):
                self.file_writer.write_metric(f'{class_name}/recall_at_{threshold}', result_dict[f'recalls_{class_name}'][i], global_step,
                                  make_sep=(class_name != 'relevant') or (i != len(ApAndAccEval.PREC_REC_CONF_THRESHOLDS) - 1))
      
        return result_dict

    @torch.no_grad()
    def eval_mse(self, logs, global_step):
        """
        Evaluate MSE
        """
        print('Computing MSE...')
        num_batches = eval_cfg.get(self.eval_mode).num_samples.mse // eval_cfg.get(self.eval_mode).batch_size
        metric_logger = MetricLogger()
        for log in logs[:num_batches]:
            metric_logger.update(mse=torch.mean(log['mse']))
        mse = metric_logger['mse'].global_avg
        self.file_writer.write_metric(f'all/mse', mse, global_step=global_step)
        return mse

    @torch.no_grad()
    def eval_clustering(self, global_step, cfg, eval_mode):
        """
        Evaluate clustering
        :return: result_dict
        """
        print('Computing clustering and few-shot linear classifiers...')
        results = ClusteringEval(cfg).eval_clustering(self.data_subset_modes, eval_mode)
        if (None, None, None) in results.values():
            results = self.ap_results_none_dict
        for name, (result_dict, img_path, few_shot_accuracy) in results.items():
            try:
                self.tb_writer.add_image(f'Clustering {name.title()}', np.array(Image.open(img_path)), global_step,
                                 dataformats='HWC')
            except:
                pass

            metrics = [f"few_shot_accuracy_with_{train_objects_per_class}" for train_objects_per_class in [1, 4, 16, 64]] \
                + ['few_shot_accuracy_cluster_nn', 'adjusted_mutual_info_score', 'adjusted_rand_score']
            combined_dict = {**result_dict, **few_shot_accuracy}
            self.file_writer.write_metrics(metrics, name, combined_dict, global_step, class_name_part_of_key=False)
            if "relevant" in name:
                self.file_writer.write_metric(f'{name}/bayes_accuracy',
                                  few_shot_accuracy[f'bayes_accuracy'], global_step)
        return results

class EvalWriter:
    def __init__(self, cfg, tb_writer: SummaryWriter, data_subset_modes, eval_mode, eval_file_path):
        self.tb_writer = tb_writer
        self.data_subset_modes = data_subset_modes
        self.eval_mode = eval_mode
        self.eval_file_path = eval_file_path

    def write_metric(self, tb_label, value, global_step, use_writer=True, make_sep=True):
        if use_writer:
            self.tb_writer.add_scalar(tb_label, value, global_step)
        self.write_to_file(value, make_sep)
    
    def write_to_file(self, value, make_sep=True):
        with open(self.eval_file_path, "a") as file:
            file.write(f'{value}' + (";" if make_sep else ""))
    
    def write_metrics(self, metrics, class_name, metrics_dict, global_step, class_name_part_of_key=True):
        for metric in metrics:
            key_string = f'{metric}_{class_name}' if class_name_part_of_key else metric
            self.write_metric(f'{class_name}/{metric}', metrics_dict[key_string], global_step)

    def write_header(self):
        columns = self.generate_header_list()
        with open(self.eval_file_path, "w") as file:
            file.write(";".join(columns))
            file.write("\n")

    def generate_header_list(self):
        columns = ['global_step']
        if 'cluster' in eval_cfg.get(self.eval_mode).metrics:
            columns.extend(self.get_cluster_header())
        if 'mse' in eval_cfg.get(self.eval_mode).metrics:
            columns.append('mse')
        if 'ap' in eval_cfg.get(self.eval_mode).metrics:
            columns.extend(self.get_ap_header())
        return columns

    def get_cluster_header(self):
        column_endings = [f"few_shot_accuracy_with_{train_objects_per_class}" for train_objects_per_class in [1, 4, 16, 64]] \
                + ['few_shot_accuracy_cluster_nn', 'adjusted_mutual_info_score', 'adjusted_rand_score']
        column_starts = self.data_subset_modes
        columns = [f"{class_name}_{column_ending}" for class_name in column_starts for column_ending in column_endings]
        columns.append(f'relevant_bayes_accuracy') # special case only for relevant
        return columns

    def get_ap_header(self):
        column_endings = [f"ap_{iou_t:.2f}" for iou_t in ApAndAccEval.AP_IOU_THRESHOLDS] + \
            ['accuracy', 'perfect', 'overcount', 'undercount', 'error_rate', 'precision', 'recall'] + \
            [f"precision_{thres:.2f}" for thres in ApAndAccEval.PREC_REC_CONF_THRESHOLDS] + \
            [f"recall_{thres:.2f}" for thres in ApAndAccEval.PREC_REC_CONF_THRESHOLDS]
        column_starts = self.data_subset_modes
        columns = [f"{class_name}_{column_ending}" for class_name in column_starts for column_ending in column_endings]
        return columns
