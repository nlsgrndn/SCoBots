import torch
import os
from tqdm import tqdm

from .eval_cfg import eval_cfg
from model.space.postprocess_latent_variables import convert_to_boxes, retrieve_latent_repr_from_logs
from PIL import Image
import os
import pickle
from .utils import flatten
from collections import Counter
from dataset.z_what import Atari_Z_What
from torch.utils.data import DataLoader


class ZWhatDataCollector:

    def __init__(self, cfg, relevant_object_hover_path):
        self.cfg = cfg
        self.bbox_examples_path = relevant_object_hover_path

    #def collect_z_what_data(self, logs, dataset, global_step,):
    #    num_samples = min(len(dataset), eval_cfg.train.num_samples.cluster)
    #    num_batches = num_samples // eval_cfg.train.batch_size
    #    batch_size = eval_cfg.train.batch_size
    #    z_encs = []
    #    z_whats = []
    #    labels_all = []
    #    labels_moving = []
    #    image_refs = []
    #    img_path = os.path.join(dataset.image_path, dataset.dataset_mode)
#
    #    # create "dataset": z_encs, z_whats, all_labels, all_labels_moving
    #    # z_enc: list of lists of tensors of shape (N, 4) where N is the number of objects in the image
    #    for i, img in enumerate(logs[:num_batches]):
    #        z_where, z_pres, z_pres_prob, z_what = retrieve_latent_repr_from_logs(img)
#
    #        z_whats.extend(z_what[z_pres])
#
    #        boxes_batch = convert_to_boxes(z_where, z_pres, z_pres_prob, with_conf=True)
#
    #        gt_bbs = dataset.get_gt_bbs(i * batch_size, (i + 1) * batch_size)
    #        gt_bbs_only_moving = gt_bbs[gt_bbs[4] == "M"]
#
    #        labels_all.extend(dataset.get_labels(gt_bbs, boxes_batch)) # assign no_label if no match with any gt object found
    #        labels_moving.extend(dataset.get_labels(gt_bbs_only_moving, boxes_batch)) # assign no_label if no match with moving gt object found
#
#
    #        # store z_encs as list of lists where each inner list contains the z_where and z_what encodings of 4 consecutive frames
    #        nr_of_consecutive_frames = 4
    #        for j in range(batch_size):
    #            datapoint_encs = []
    #            for k in range(nr_of_consecutive_frames):
    #                index = j * nr_of_consecutive_frames + k
    #                z_wr, z_pr, z_wt = z_where[index], z_pres_prob[index], z_what[index]
    #                z_pr = z_pr.squeeze() > 0.5
    #                datapoint_encs.append(torch.cat([z_wr[z_pr], z_wt[z_pr]], dim=1))
    #            z_encs.append(datapoint_encs)
#
    #        # store image references of bounding boxes
    #        if self.cfg.save_relevant_objects:
    #            self.save_relevant_objects_as_images(global_step, image_refs, batch_size, img_path, i, z_where, z_pres)
#
    #     
#
    #    labels_relevant_idx, labels_relevant = dataset.to_relevant(labels_moving) # get idx of labels that are not no_label
#
    #    if len(z_whats) == 0 or len(labels_relevant) == 0:
    #        ret_value = None, None, None, None, None
    #        data = {
    #            "all": ret_value,
    #            "moving": ret_value,
    #            "relevant": ret_value,
    #        }
    #        return data, None, None
    #    
    #    z_whats = torch.stack(z_whats).detach().cpu()
    #    z_whats_all = z_whats
    #    z_whats_moving = z_whats
    #    z_whats_relevant = z_whats[flatten(labels_relevant_idx)]
#
    #    labels_relevant_unflattened = labels_relevant
    #    labels_all = flatten(labels_all)
    #    labels_moving = flatten(labels_moving)
    #    labels_relevant = flatten(labels_relevant)
#
    #    # data for kalman filter
    #    z_encs_relevant = [[enc[rel_idx] for enc, rel_idx in zip(enc_seq, rel_seq)] for enc_seq, rel_seq in zip(z_encs, labels_relevant_idx)]
#
    #    data = {
    #        "all": (z_whats_all, labels_all),
    #        "moving": (z_whats_moving, labels_moving),
    #        "relevant": (z_whats_relevant, labels_relevant),
    #    }
    #    for key in data:
    #        data[key] = self.prepare_data(*data[key])
#
    #    # save relevant objects
    #    if self.cfg.save_relevant_objects:
    #        self.save_relevant_objects_as_pickle(global_step, labels_relevant_idx, z_whats_relevant, labels_relevant, image_refs)
#
    #    return data, z_encs_relevant, labels_relevant_unflattened
    
    def collect_z_what_data(self, logs, dataset, global_step, cfg):
        data_subset_modes = ['all', 'relevant'] # TODO configure somewhere else
        dataset_mode = "test" # TODO configure somewhere else

        data = {}
        for boxes_subset in data_subset_modes:
            atari_z_what_dataset = Atari_Z_What(cfg, dataset_mode, boxes_subset, return_keys = ["z_whats_pres_s", "gt_labels_for_pred_boxes"])
            atari_z_what_dataloader = DataLoader(atari_z_what_dataset, batch_size=1, shuffle=False, num_workers=0) #batch_size must be 1
            z_whats = []
            labels = []
            for batch in atari_z_what_dataloader:
                curr_z_whats = batch["z_whats_pres_s"]
                curr_labels = batch["gt_labels_for_pred_boxes"]
                z_whats.extend([curr_z_whats[i][0] for i in range(len(curr_z_whats))]) #[0] because batch_size = 1 and dataloader implicitly adds a batch dimension
                labels.extend([curr_labels[i][0] for i in range(len(curr_labels))]) #[0] because batch_size = 1 and dataloader implicitly adds a batch dimension
            # z_whats and labels are lists of tensors of shape
            z_whats = torch.cat(z_whats, dim=0)
            labels= torch.cat(labels, dim=0)
            labels = labels.squeeze(-1)
            data[boxes_subset] = (z_whats, labels)

        for key in data:
            data[key] = self.prepare_data(*data[key])

        # save relevant objects
        #if self.cfg.save_relevant_objects:
        #    self.save_relevant_objects_as_pickle(global_step, labels_relevant_idx, z_whats_relevant, labels_relevant, image_refs)

        return data
    
    def prepare_data(self, z_what, labels,):
        import ipdb; ipdb.set_trace()
        c = Counter(labels.tolist() if labels is not None else [])
        relevant_labels = list(c.keys())

        # Filter out the irrelevant labels
        z_what, labels = self.only_keep_relevant_data(z_what, labels, relevant_labels)
        # Split the data into train and test
        train_x, train_y, test_x, test_y = self.train_test_split(z_what, labels, train_portion=0.9)

        if len(c) < 2 or len(torch.unique(train_y)) < 2:
            return None, None, None, None, None
        
        return relevant_labels, test_x, test_y, train_x, train_y
        
    def train_test_split(self, z_what, labels, train_portion=0.9):
        nb_sample = int(train_portion * len(labels))
        train_x = z_what[:nb_sample]
        train_y = labels[:nb_sample]
        test_x = z_what[nb_sample:]
        test_y = labels[nb_sample:]
        return train_x, train_y, test_x, test_y

    def only_keep_relevant_data(self, z_what, labels, relevant_labels):
        relevant_mask = torch.zeros(labels.shape, dtype=torch.bool)
        for rl in relevant_labels:
            relevant_mask |= labels == rl
        return z_what[relevant_mask], labels[relevant_mask]

    def save_relevant_objects_as_images(self, global_step, image_refs, batch_size, img_path, i, z_where, z_pres):
        for idx, (sel, bbs) in enumerate(tqdm(zip(z_pres, z_where), total=len(z_pres))):
            for obj_idx, bb in enumerate(bbs[sel]):
                image = Image.open(os.path.join(img_path, f'{i * batch_size + idx // 4:05}_{idx % 4}.png'))
                width, height, center_x, center_y = bb.tolist()
                center_x = (center_x + 1.0) / 2.0 * 128
                center_y = (center_y + 1.0) / 2.0 * 128
                bb = (int(center_x - width * 128),
                              int(center_y - height * 128),
                              int(center_x + width * 128),
                              int(center_y + height * 128))
                try:
                    cropped = image.crop(bb)
                    cropped.save(f'{self.bbox_examples_path}/img/'
                                         f'gs{global_step:06}_{i * batch_size + idx // 4:05}_{idx % 4}_obj{obj_idx}.png')
                except:
                    image.save(f'{self.bbox_examples_path}/img/'
                                       f'gs{global_step:06}_{i * batch_size + idx // 4:05}_{idx % 4}_obj{obj_idx}.png')
                new_image_path = f'gs{global_step:06}_{i * batch_size + idx // 4:05}_{idx % 4}_obj{obj_idx}.png'
                image_refs.append(new_image_path)
        return image_refs
    
    def save_relevant_objects_as_pickle(self, global_step, labels_relevant_idx, z_whats_relevant, labels_relevant, image_refs):
        image_refs_relevant = [image_refs[idx] for idx, yes in enumerate(labels_relevant_idx) if yes]
        # Save relevant objects data to pickle file
        with open(f'{self.bbox_examples_path}/relevant_objects_{global_step:06}.pkl', 'wb') as output_file:
            relevant_objects_data = {
                'z_what': z_whats_relevant,
                'labels': labels_relevant,
                'image_refs': image_refs_relevant,
            }
            pickle.dump(relevant_objects_data, output_file, pickle.DEFAULT_PROTOCOL)