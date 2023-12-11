import torch
import os
from tqdm import tqdm

from .eval_cfg import eval_cfg
from model.space.postprocess_latent_variables import convert_to_boxes, retrieve_latent_repr_from_logs
from .kalman_filter import classify_encodings
from PIL import Image
from .classify_z_what import ZWhatEvaluator
import os
import pickle
from .utils import flatten
from collections import Counter

class ClusteringEval:

    def __init__(self, cfg, relevant_object_hover_path, indices = None):
        self.cfg = cfg
        self.relevant_object_hover_path = relevant_object_hover_path
        self.indices = indices # 'The relevant objects by their index, e.g. \"0,1\" for Pacman and Sue')
    
    def collect_data(self, logs, dataset, global_step,):
        num_samples = min(len(dataset), eval_cfg.train.num_samples.cluster)
        num_batches = num_samples // eval_cfg.train.batch_size
        batch_size = eval_cfg.train.batch_size
        z_encs = []
        z_whats = []
        labels_all = []
        labels_moving = []
        image_refs = []
        img_path = os.path.join(dataset.image_path, dataset.dataset_mode)

        # create "dataset": z_encs, z_whats, all_labels, all_labels_moving
        # z_enc: list of lists of tensors of shape (N, 4) where N is the number of objects in the image
        for i, img in enumerate(logs[:num_batches]):
            z_where, z_pres, z_pres_prob, z_what = retrieve_latent_repr_from_logs(img)
            if not (0.05 <= z_pres.sum() / batch_size <= 60 * 4):
                continue

            z_whats.extend(z_what[z_pres])

            boxes_batch = convert_to_boxes(z_where, z_pres, z_pres_prob, with_conf=True)
            labels_all.extend(dataset.get_labels(i * batch_size, (i + 1) * batch_size, boxes_batch))
            labels_moving.extend(dataset.get_labels_moving(i * batch_size, (i + 1) * batch_size, boxes_batch))

            # store z_encs as list of lists where each inner list contains the z_where and z_what encodings of 4 consecutive frames
            nr_of_consecutive_frames = 4
            for j in range(batch_size):
                datapoint_encs = []
                for k in range(nr_of_consecutive_frames):
                    index = j * nr_of_consecutive_frames + k
                    z_wr, z_pr, z_wt = z_where[index], z_pres_prob[index], z_what[index]
                    z_pr = z_pr.squeeze() > 0.5
                    datapoint_encs.append(torch.cat([z_wr[z_pr], z_wt[z_pr]], dim=1))
                z_encs.append(datapoint_encs)

            # store image references of bounding boxes
            if self.cfg.save_relevant_objects:
                self.save_relevant_objects_as_images(global_step, image_refs, batch_size, img_path, i, z_where, z_pres)

         

        labels_relevant_idx, labels_relevant = dataset.to_relevant(labels_moving)

        if len(z_whats) == 0 or len(labels_relevant) == 0:
            ret_value = None, None, None, None, None
            data = {
                "all": ret_value,
                "moving": ret_value,
                "relevant": ret_value,
            }
            return data, None, None
        
        z_whats = torch.stack(z_whats).detach().cpu()
        z_whats_all = z_whats
        z_whats_moving = z_whats
        z_whats_relevant = z_whats[flatten(labels_relevant_idx)]

        labels_relevant_unflattened = labels_relevant
        labels_all = flatten(labels_all)
        labels_moving = flatten(labels_moving)
        labels_relevant = flatten(labels_relevant)

        # data for kalman filter
        z_encs_relevant = [[enc[rel_idx] for enc, rel_idx in zip(enc_seq, rel_seq)] for enc_seq, rel_seq in zip(z_encs, labels_relevant_idx)]

        data = {
            "all": (z_whats_all, labels_all),
            "moving": (z_whats_moving, labels_moving),
            "relevant": (z_whats_relevant, labels_relevant),
        }
        for key in data:
            data[key] = self.prepare_data(*data[key])

        # save relevant objects
        if self.cfg.save_relevant_objects:
            self.save_relevant_objects_as_pickle(global_step, labels_relevant_idx, z_whats_relevant, labels_relevant, image_refs)

        return data, z_encs_relevant, labels_relevant_unflattened
    
    def prepare_data(self, z_what, labels,):
        c = Counter(labels.tolist() if labels is not None else [])
        if self.cfg.train.log:
            print("Distribution of matched labels:", c)
        # Initialization stuff
        relevant_labels = [int(part) for part in self.indices.split(',')] if self.indices else list(c.keys())

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

    def only_keep_relevant_data(sefl, z_what, labels, relevant_labels):
        relevant = torch.zeros(labels.shape, dtype=torch.bool)
        for rl in relevant_labels:
            relevant |= labels == rl
        return z_what[relevant], labels[relevant]
    
    @torch.no_grad()
    def eval_clustering(self, logs, dataset, global_step,):
        """
        Evaluate clustering metrics

        :param logs: results from applying the model
        :param dataset: dataset
        :param global_step: gradient step number
        :return metrics: for all classes of evaluation many metrics describing the clustering,
            based on different ground truths
        """
        data, z_encs_relevant, labels_relevant_unflattened = \
            self.collect_data(logs, dataset, global_step,)
        
        results = {}
        for key in ['all', 'moving', 'relevant']:
            relevant_labels, test_x, test_y, train_x, train_y = data[key]
            objects = ZWhatEvaluator(self.cfg, title= key,).evaluate_z_what(train_x, train_y, test_x, test_y, relevant_labels)
            if key == 'relevant' and objects[2] is not None:
                # add bayes accuracy metric for relevant objects based on Kalman filter
                bayes_accuracy = classify_encodings(self.cfg, z_encs_relevant, labels_relevant_unflattened)
                objects[2]['bayes_accuracy'] = bayes_accuracy
            results[key] = objects
        return results

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
                    cropped.save(f'{self.relevant_object_hover_path}/img/'
                                         f'gs{global_step:06}_{i * batch_size + idx // 4:05}_{idx % 4}_obj{obj_idx}.png')
                except:
                    image.save(f'{self.relevant_object_hover_path}/img/'
                                       f'gs{global_step:06}_{i * batch_size + idx // 4:05}_{idx % 4}_obj{obj_idx}.png')
                new_image_path = f'gs{global_step:06}_{i * batch_size + idx // 4:05}_{idx % 4}_obj{obj_idx}.png'
                image_refs.append(new_image_path)
        return image_refs
    
    def save_relevant_objects_as_pickle(self, global_step, labels_relevant_idx, z_whats_relevant, labels_relevant, image_refs):
        image_refs_relevant = [image_refs[idx] for idx, yes in enumerate(labels_relevant_idx) if yes]
        # Save relevant objects data to pickle file
        with open(f'{self.relevant_object_hover_path}/relevant_objects_{global_step:06}.pkl', 'wb') as output_file:
            relevant_objects_data = {
                'z_what': z_whats_relevant,
                'labels': labels_relevant,
                'image_refs': image_refs_relevant,
            }
            pickle.dump(relevant_objects_data, output_file, pickle.DEFAULT_PROTOCOL)

    