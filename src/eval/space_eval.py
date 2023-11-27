from eval.clustering_eval import ClusteringEval
from eval.ap_and_acc_eval import ApAndAccEval
from utils import MetricLogger
import numpy as np
import torch
import os
import sys
from datetime import datetime
from torch.utils.data import Subset, DataLoader
import os.path as osp
from tqdm import tqdm
from .eval_cfg import eval_cfg
from .ap import read_boxes, convert_to_boxes, compute_ap, compute_counts, compute_prec_rec
from .kalman_filter import classify_encodings
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from .classify_z_what import evaluate_z_what
import PIL
from torchvision.utils import draw_bounding_boxes as draw_bb
import os
import pprint
import pickle
from .utils import flatten


class SpaceEval:
    def __init__(self):
        self.relevant_object_hover_path = None
        self.eval_file = None
        self.eval_file_path = None
        self.first_eval = True

    # @torch.no_grad()
    # def test_eval(self, model, testset, bb_path, device, evaldir, info, global_step, cfg):
    #     losses, logs = self.apply_model(testset, device, model, global_step)
    #     result_dict = self.eval_ap_and_acc(logs, testset, bb_path)
    #     clustering_result_dict = self.eval_clustering(logs, testset, global_step, cfg)
    #     os.makedirs(evaldir, exist_ok=True)
    #     path = osp.join(evaldir, 'results_{}.json'.format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    #     self.save_to_json(result_dict, path, info)
    #     self.print_result(result_dict, [sys.stdout, open('./results.txt', 'w')])

    def write_metric(self, writer, tb_label, value, global_step, use_writer=True, make_sep=True):
        if use_writer:
            writer.add_scalar(tb_label, value, global_step)
        self.eval_file.write(f'{value};' if make_sep else f'{value}')

    def write_header(self):
        columns = ['global_step']
        if 'cluster' in eval_cfg.train.metrics:
            for class_name in ['all', 'moving', 'relevant']:
                for training_objects_per_class in [1, 4, 16, 64]:
                    columns.append(f'{class_name}_few_shot_accuracy_with_{training_objects_per_class}')
                columns.append(f'{class_name}_few_shot_accuracy_cluster_nn')
                columns.append(f'{class_name}_adjusted_mutual_info_score')
                columns.append(f'{class_name}_adjusted_rand_score')
            columns.append(f'relevant_bayes_accuracy')
        if 'mse' in eval_cfg.train.metrics:
            columns.append('mse')
        if 'ap' in eval_cfg.train.metrics:
            for class_name in ['all', 'moving', 'relevant']:
                for iou_t in np.linspace(0.1, 0.9, 9):
                    columns.append(f'{class_name}_ap_{iou_t:.2f}')
                columns.append(f'{class_name}_accuracy')
                columns.append(f'{class_name}_perfect')
                columns.append(f'{class_name}_overcount')
                columns.append(f'{class_name}_undercount')
                columns.append(f'{class_name}_error_rate')
                columns.append(f'{class_name}_precision')
                columns.append(f'{class_name}_recall')

        with open(self.eval_file_path, "w") as file:
            file.write(";".join(columns))
            file.write("\n")

    @torch.no_grad()
    # @profile
    def train_eval(self, model, valset, bb_path, writer, global_step, device, checkpoint, checkpointer, cfg):
        """
        Evaluation during training. This includes:
            - mse evaluated on validation set
            - ap and accuracy evaluated on validation set
            - cluster metrics evaluated on validation set
        :return:
        """
        if self.first_eval:
            self.first_eval = False
            self.eval_file_path = f'{cfg.logdir}/{cfg.exp_name}/metrics.csv'
            self.relevant_object_hover_path = f'{cfg.logdir}/{cfg.exp_name}/hover'
            if os.path.exists(self.eval_file_path):
                os.remove(self.eval_file_path)
            os.makedirs(self.relevant_object_hover_path, exist_ok=True)
            os.makedirs(self.relevant_object_hover_path + "Img", exist_ok=True)
            self.write_header()

        _, logs = self.apply_model(valset, device, model, global_step)
        self.core_eval_code(valset, bb_path, writer, global_step, checkpoint, checkpointer, cfg, logs, save_best = True)



    @torch.no_grad()
    def test_eval(self, model, testset, bb_path, writer, global_step, device, checkpoint, checkpointer, cfg):
        """
        Evaluation during testing. This includes:
            - mse evaluated on test set
            - ap and accuracy evaluated on test set
            - cluster metrics evaluated on test set
        :return:
        """
        # make checkpoint test dir
        chpt_dir_save = checkpointer.checkpointdir
        model_str = model.arch_type if hasattr(model, "arch_type") else model.module.arch_type # access string for cpu else for gpu
        checkpointer.checkpointdir = chpt_dir_save.replace("/eval/", "/test_eval/") + f"_{model_str}"
        os.makedirs(checkpointer.checkpointdir, exist_ok=True)

        # path and file handling
        efp_save = self.eval_file_path
        rohp_save = self.relevant_object_hover_path
        self.eval_file_path = f'../final_test_results/{cfg.exp_name}_{model_str}_seed{cfg.seed}_metrics.csv'
        self.relevant_object_hover_path = f'../final_test_results/{cfg.exp_name}/hover'
        if os.path.exists(self.eval_file_path):
            os.remove(self.eval_file_path)
        os.makedirs(self.relevant_object_hover_path, exist_ok=True)
        os.makedirs(f'../final_test_results/{cfg.exp_name}/hover', exist_ok=True)
        os.makedirs(os.path.join(self.relevant_object_hover_path, "img"), exist_ok=True)

        # write header to csv file
        self.write_header()

        # apply model to test set and logs
        _, logs = self.apply_model(testset, device, model, global_step)

        # evaluate test set using core evaluation code
        self.core_eval_code(testset, bb_path, writer, global_step, checkpoint, checkpointer, cfg, logs, save_best = False)
        print(f"Saved results in {self.eval_file_path}")

        # reset paths to original values
        self.eval_file_path = efp_save
        self.relevant_object_hover_path = rohp_save
        checkpointer.checkpointdir = chpt_dir_save

    def core_eval_code(self, valset, bb_path, writer, global_step, checkpoint, checkpointer, cfg, logs, save_best):
        with open(self.eval_file_path, "a") as self.eval_file:
            self.write_metric(None, None, global_step, global_step, use_writer=False)
            if 'cluster' in eval_cfg.train.metrics:
                results = self.train_eval_clustering(logs, valset, writer, global_step, cfg)
                if cfg.train.log:
                    pp = pprint.PrettyPrinter(depth=2)
                    for res in results:
                        print("Cluster Result:")
                        pp.pprint(results[res])
                if save_best:
                    checkpointer.save_best('rand_score_relevant',
                                       results['relevant'][0]['adjusted_rand_score'],
                                       checkpoint, min_is_better=False)
            if 'mse' in eval_cfg.train.metrics:
                mse = self.train_eval_mse(logs, writer, global_step)
                print("MSE result: ", mse)
            if 'ap' in eval_cfg.train.metrics:
                results = self.train_eval_ap_and_acc(logs, valset, bb_path, writer, global_step)
                if save_best:
                    checkpointer.save_best('accuracy', results['accuracy_relevant'], checkpoint, min_is_better=False)
                if cfg.train.log:
                    results = {k2: v2[len(v2) // 4] if isinstance(v2, list) or isinstance(v2, np.ndarray) else v2 for
                               k2, v2, in
                               results.items()}
                    pp = pprint.PrettyPrinter(depth=2)
                    print("AP Result:")
                    pp.pprint({k: v for k, v in results.items() if "iou" not in k})
            self.eval_file.write("\n")

    @torch.no_grad()
    def apply_model(self, dataset, device, model, global_step, use_global_step=False):
        print('Applying the model for evaluation...')
        model.eval()
        if eval_cfg.train.num_samples:
            num_samples = min(len(dataset),max(eval_cfg.train.num_samples.values()))
        else:
            num_samples = len(dataset)
        # eval_cfg.train.batch_size = 4 Why was this manually overwritten to 4 here?
        batch_size = eval_cfg.train.batch_size
        num_workers = eval_cfg.train.num_workers
        data_subset = Subset(dataset, indices=range(num_samples))
        dataloader = DataLoader(data_subset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        losses = []
        logs = []
        with torch.no_grad():
            for imgs, motion, motion_z_pres, motion_z_where in dataloader:
                imgs = imgs.to(device)
                motion = None
                motion_z_pres = None
                motion_z_where = None
                # motion = motion.to(device)
                # motion_z_pres = motion_z_pres.to(device)
                # motion_z_where = motion_z_where.to(device)
                loss, log = model(imgs, motion, motion_z_pres, motion_z_where,
                                  global_step if use_global_step else 1000000000)
                for key in ['imgs', 'y', 'log_like', 'loss', 'fg', 'z_pres_prob_pure',
                            'prior_z_pres_prob', 'o_att', 'alpha_att_hat', 'alpha_att', 'alpha_map', 'alpha_map_pure',
                            'importance_map_full_res_norm', 'kl_z_what', 'kl_z_pres', 'kl_z_scale', 'kl_z_shift',
                            'kl_z_depth', 'kl_z_where', 'comps', 'masks', 'bg', 'kl_bg']:
                    del log[key]
                losses.append(loss)
                logs.append(log)
        model.train()
        return losses, logs

    @torch.no_grad()
    def train_eval_ap_and_acc(self, logs, valset, bb_path, writer: SummaryWriter, global_step):
        """
        Evaluate ap and accuracy during training
        :return: result_dict
        """
        result_dict = ApAndAccEval().eval_ap_and_acc(logs, valset, bb_path)
        for class_name in ['all', 'moving', 'relevant']:
            APs = result_dict[f'APs_{class_name}']
            iou_thresholds = result_dict[f'iou_thresholds_{class_name}']
            accuracy = result_dict[f'accuracy_{class_name}']
            perfect = result_dict[f'perfect_{class_name}']
            overcount = result_dict[f'overcount_{class_name}']
            undercount = result_dict[f'undercount_{class_name}']
            error_rate = result_dict[f'error_rate_{class_name}']
            precision = result_dict[f'precision_{class_name}']
            recall = result_dict[f'recall_{class_name}']

            for ap in APs:
                self.write_metric(None, 'ignored', ap, global_step, use_writer=False)
            for ap, thres in zip(APs[1::4], iou_thresholds[1::4]):
                writer.add_scalar(f'val_aps_{class_name}/ap_{thres:.1}', ap, global_step)
            writer.add_scalar(f'{class_name}/ap_avg_0.5', APs[len(APs) // 2], global_step)
            writer.add_scalar(f'{class_name}/ap_avg_up', np.mean(APs[len(APs) // 2:]), global_step)
            writer.add_scalar(f'{class_name}/ap_avg', np.mean(APs), global_step)
            self.write_metric(writer, f'{class_name}/accuracy', accuracy, global_step)
            self.write_metric(writer, f'{class_name}/perfect', perfect, global_step)
            self.write_metric(writer, f'{class_name}/overcount', overcount, global_step)
            self.write_metric(writer, f'{class_name}/undercount', undercount, global_step)
            self.write_metric(writer, f'{class_name}/error_rate', error_rate, global_step)
            self.write_metric(writer, f'{class_name}/precision', precision, global_step)
            self.write_metric(writer, f'{class_name}/recall', recall, global_step,
                              make_sep=class_name != 'relevant')
        return result_dict

    @torch.no_grad()
    def train_eval_mse(self, logs, writer, global_step):
        """
        Evaluate MSE during training
        """
        print('Computing MSE...')
        num_batches = eval_cfg.train.num_samples.mse // eval_cfg.train.batch_size
        metric_logger = MetricLogger()
        for log in logs[:num_batches]:
            metric_logger.update(mse=torch.mean(log['mse']))
        mse = metric_logger['mse'].global_avg
        self.write_metric(writer, f'all/mse', mse, global_step=global_step)
        return mse

    @torch.no_grad()
    def train_eval_clustering(self, logs, valset, writer: SummaryWriter, global_step, cfg):
        """
        Evaluate clustering during training
        :return: result_dict
        """
        print('Computing clustering and few-shot linear classifiers...')
        results = ClusteringEval(self.relevant_object_hover_path).eval_clustering(logs, valset, global_step, cfg)
        for name, (result_dict, img_path, few_shot_accuracy) in results.items():
            try:
                writer.add_image(f'Clustering PCA {name.title()}', np.array(Image.open(img_path)), global_step,
                                 dataformats='HWC')
            except:
                pass
            for train_objects_per_class in [1, 4, 16, 64]:
                self.write_metric(writer, f'{name}/few_shot_accuracy_with_{train_objects_per_class}',
                                  few_shot_accuracy[f'few_shot_accuracy_with_{train_objects_per_class}'], global_step)
            self.write_metric(writer, f'{name}/few_shot_accuracy_cluster_nn',
                              few_shot_accuracy[f'few_shot_accuracy_cluster_nn'], global_step)
            self.write_metric(writer, f'{name}/adjusted_mutual_info_score',
                              result_dict[f'adjusted_mutual_info_score'], global_step)
            self.write_metric(writer, f'{name}/adjusted_rand_score',
                              result_dict[f'adjusted_rand_score'], global_step)
            if "relevant" in name:
                self.write_metric(writer, f'{name}/bayes_accuracy',
                                  few_shot_accuracy[f'bayes_accuracy'], global_step)
        return results

    #def save_to_json(self, result_dict, json_path, info):
    #    """
    #    Save evaluation results to json file
    #
    #    :param result_dict: a dictionary
    #    :param json_path: checkpointdir
    #    :param info: any other thing you want to save
    #    :return:
    #    """
    #    from collections import OrderedDict
    #    import json
    #    from datetime import datetime
    #    tosave = OrderedDict([
    #        ('date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    #        ('info', info),
    #    ])
    #    for metric in ['APs', 'accuracy', 'error_rate', 'iou_thresholds', 'overcount', 'perfect', 'precision', 'recall', 'undercount']:
    #        if hasattr(result_dict[f'{metric}_all'], '__iter__'):
    #            tosave[metric] = list(result_dict[f'{metric}_all'])
    #            tosave[f'{metric}_avg'] = np.mean(result_dict[f'{metric}_all'])
    #            tosave[f'{metric}_relevant'] = list(result_dict[f'{metric}_relevant'])
    #            tosave[f'{metric}_relevant_avg'] = np.mean(result_dict[f'{metric}_relevant'])
    #            tosave[f'{metric}_moving'] = list(result_dict[f'{metric}_moving'])
    #            tosave[f'{metric}_moving_avg'] = np.mean(result_dict[f'{metric}_moving'])
    #        else:
    #            tosave[metric] = result_dict[f'{metric}_all']
    #            tosave[f'{metric}_relevant'] = result_dict[f'{metric}_relevant']
    #            tosave[f'{metric}_moving'] = result_dict[f'{metric}_moving']
    #    with open(json_path, 'w') as f:
    #        json.dump(tosave, f, indent=2)
    #
    #    print(f'Results have been saved to {json_path}.')

    #def print_result(self, result_dict, files):
    #    for suffix in ['all', 'relevant', 'moving']:
    #        APs = result_dict[f'APs_{suffix}']
    #        iou_thresholds = result_dict[f'iou_thresholds_{suffix}']
    #        accuracy = result_dict[f'accuracy_{suffix}']
    #        perfect = result_dict[f'perfect_{suffix}']
    #        overcount = result_dict[f'overcount_{suffix}']
    #        undercount = result_dict[f'undercount_{suffix}']
    #        error_rate = result_dict[f'error_rate_{suffix}']
    #        for file in files:
    #            print('\n' + '-' * 10 + f'metrics on {suffix} data points' + '-' * 10, file=file)
    #            print('{:^15} {:^15}'.format('IoU threshold', 'AP'), file=file)
    #            print('{:15} {:15}'.format('-' * 15, '-' * 15), file=file)
    #            for thres, ap in zip(iou_thresholds, APs):
    #                print('{:<15.2} {:<15.4}'.format(thres, ap), file=file)
    #            print('{:15} {:<15.4}'.format('Average:', np.mean(APs)), file=file)
    #            print('{:15} {:15}'.format('-' * 15, '-' * 15), file=file)
    #
    #            print('{:15} {:<15}'.format('Perfect:', perfect), file=file)
    #            print('{:15} {:<15}'.format('Overcount:', overcount), file=file)
    #            print('{:15} {:<15}'.format('Undercount:', undercount), file=file)
    #            print('{:15} {:<15.4}'.format('Accuracy:', accuracy), file=file)
    #            print('{:15} {:<15.4}'.format('Error rate:', error_rate), file=file)
    #            print('{:15} {:15}'.format('-' * 15, '-' * 15), file=file)

    # def save_best(self, evaldir, metric_name, value, checkpoint, checkpointer, min_is_better):
    #     metric_file = os.path.join(evaldir, f'best_{metric_name}.json')
    #     checkpoint_file = os.path.join(evaldir, f'best_{metric_name}.pth')
    #
    #     now = datetime.now()
    #     log = {
    #         'name': metric_name,
    #         'value': float(value),
    #         'date': now.strftime("%Y-%m-%d %H:%M:%S"),
    #         'global_step': checkpoint[-1]
    #     }
    #
    #     if not os.path.exists(metric_file):
    #         dump = True
    #     else:
    #         with open(metric_file, 'r') as f:
    #             previous_best = json.load(f)
    #         if not math.isfinite(log['value']):
    #             dump = True
    #         elif (min_is_better and log['value'] < previous_best['value']) or (
    #                 not min_is_better and log['value'] > previous_best['value']):
    #             dump = True
    #         else:
    #             dump = False
    #     if dump:
    #         with open(metric_file, 'w') as f:
    #             json.dump(log, f)
    #         checkpointer.save(checkpoint_file, *checkpoint)
