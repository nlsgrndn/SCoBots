import numpy as np
import torch
from .eval_cfg import eval_cfg
from .ap import read_boxes, convert_to_boxes, compute_ap, compute_counts, compute_prec_rec

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
        boxes_gts = {k: v for k, v in zip(boxes_gt_types, read_boxes(bb_path, indices=indices))}
        boxes_pred = []
        boxes_relevant = []

        num_batches = min(len(dataset), eval_cfg.train.num_samples.cluster) // batch_size #eval_cfg.train.num_samples.cluster // eval_cfg.train.batch_size

        for img in logs[:num_batches]:
            z_where, z_pres_prob, z_what = img['z_where'], img['z_pres_prob'], img['z_what']
            z_where = z_where.detach().cpu()
            z_pres_prob = z_pres_prob.detach().cpu().squeeze()
            z_pres = z_pres_prob > 0.5
            boxes_batch = convert_to_boxes(z_where, z_pres, z_pres_prob, with_conf=True)
            boxes_relevant.extend(dataset.filter_relevant_boxes(boxes_batch, boxes_gts['all']))
            boxes_pred.extend(boxes_batch)

        # print('Drawing bounding boxes for eval...')
        # for i in range(4):
        #     for idx, pred, rel, gt, gt_m, gt_r in zip(indices, boxes_pred[i::4], boxes_relevant[i::4], *(gt[i::4] for gt in boxes_gts.values())):
        #         if len(pred) == len(rel):
        #             continue
        #         pil_img = Image.open(f'{rgb_folder_src}/{idx:05}_{i}.png').convert('RGB')
        #         pil_img = pil_img.resize((128, 128), PIL.Image.BILINEAR)
        #         image = np.array(pil_img)
        #         torch_img = torch.from_numpy(image).permute(2, 1, 0)
        #         pred_tensor = torch.FloatTensor(pred) * 128
        #         pred_tensor = torch.index_select(pred_tensor, 1, torch.LongTensor([0, 2, 1, 3]))
        #         rel_tensor = torch.FloatTensor(rel) * 128
        #         rel_tensor = torch.index_select(rel_tensor, 1, torch.LongTensor([0, 2, 1, 3]))
        #         gt_tensor = torch.FloatTensor(gt) * 128
        #         gt_tensor = torch.index_select(gt_tensor, 1, torch.LongTensor([0, 2, 1, 3]))
        #         gt_m_tensor = torch.FloatTensor(gt_m) * 128
        #         gt_m_tensor = torch.index_select(gt_m_tensor, 1, torch.LongTensor([0, 2, 1, 3]))
        #         gt_r_tensor = torch.FloatTensor(gt_r) * 128
        #         gt_r_tensor = torch.index_select(gt_r_tensor, 1, torch.LongTensor([0, 2, 1, 3]))
        #         bb_img = torch_img
        #         bb_img = draw_bb(torch_img, gt_tensor, colors=["red"] * len(gt_tensor))
        #         # bb_img = draw_bb(bb_img, gt_m_tensor, colors=["blue"] * len(gt_m_tensor))
        #         # bb_img = draw_bb(bb_img, gt_r_tensor, colors=["yellow"] * len(gt_r_tensor))
        #         bb_img = draw_bb(bb_img, pred_tensor, colors=["green"] * len(pred_tensor))
        #         # bb_img = draw_bb(bb_img, rel_tensor, colors=["white"] * len(rel_tensor))
        #         bb_img = Image.fromarray(bb_img.permute(2, 1, 0).numpy())
        #         bb_img.save(f'{rgb_folder}/gt_moving_p{idx:05}_{i}.png')
        #         print(f'{rgb_folder}/gt_moving_{idx:05}.png')

        result = {}
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
            precision, recall = compute_prec_rec(boxes, gt)
            result[f'APs_{gt_name}'] = aps
            result[f'precision_{gt_name}'] = precision
            result[f'recall_{gt_name}'] = recall
        return result