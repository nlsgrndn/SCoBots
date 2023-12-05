import torch
import os
from tqdm import tqdm
from .eval_cfg import eval_cfg
from .ap import convert_to_boxes
from .kalman_filter import classify_encodings
from PIL import Image
from .classify_z_what import ZWhatEvaluator
import os
import pickle
from .utils import flatten, retrieve_latent_repr_from_logs

class ClusteringEval:

    def __init__(self, relevant_object_hover_path):
        self.relevant_object_hover_path = relevant_object_hover_path
    
    def collect_data(self, logs, dataset, global_step, cfg):
        z_encs = []
        z_whats = []
        all_labels = []
        all_labels_moving = []
        image_refs = []
        batch_size = eval_cfg.train.batch_size
        img_path = os.path.join(dataset.image_path, dataset.dataset_mode)

        # create "dataset": z_encs, z_whats, all_labels, all_labels_moving
        # z_enc: list of lists of tensors of shape (N, 4) where N is the number of objects in the image
        for i, img in enumerate(logs):

            # retrieve logged latent variables
            z_where, z_pres, z_pres_prob, z_what = retrieve_latent_repr_from_logs(img)
            if not (0.05 <= z_pres.sum() / batch_size <= 60 * 4):
                continue
            # store image references with bounding boxes
            if cfg.save_relevant_objects:
                self.save_relevant_objects(global_step, image_refs, batch_size, img_path, i, z_where, z_pres)

            boxes_batch = convert_to_boxes(z_where, z_pres, z_pres_prob, with_conf=True)
            z_whats.extend(z_what[z_pres])

            # collect what and where latent variables
            # 4 because 4 consecutive images in the batch
            for j in range(len(z_pres_prob) // 4):
                datapoint_encs = []
                for k in range(4):
                    z_wr, z_pr, z_wt = z_where[j * 4 + k], z_pres_prob[j * 4 + k], z_what[j * 4 + k]
                    z_pr = z_pr.squeeze() > 0.5
                    datapoint_encs.append(torch.cat([z_wr[z_pr], z_wt[z_pr]], dim=1))
                z_encs.append(datapoint_encs)

            all_labels.extend(dataset.get_labels(i * batch_size, (i + 1) * batch_size, boxes_batch))
            all_labels_moving.extend(dataset.get_labels_moving(i * batch_size, (i + 1) * batch_size, boxes_batch))

        return z_encs, z_whats, all_labels, all_labels_moving, image_refs

    # @profile
    @torch.no_grad()
    def eval_clustering(self, logs, dataset, global_step, cfg):
        """
        Evaluate clustering metrics

        :param logs: results from applying the model
        :param dataset: dataset
        :param cfg: config
        :param global_step: gradient step number
        :return metrics: for all classes of evaluation many metrics describing the clustering,
            based on different ground truths
        """
        z_encs, z_whats, all_labels, all_labels_moving, image_refs = self.collect_data(logs, dataset, global_step, cfg)

        args = {'method': 't-SNE', 'indices': None, 'dim': 2, 'edgecolors': False} # method was set to k-means before
        if z_whats:
            z_whats = torch.stack(z_whats).detach().cpu()
            all_labels_relevant_idx, all_labels_relevant = dataset.to_relevant(all_labels_moving)
            z_whats_relevant = z_whats[flatten(all_labels_relevant_idx)]

            # call evaluate_z_what for all, moving and relevant objects
            all_objects = ZWhatEvaluator(cfg, title= "all", **args).evaluate_z_what(z_whats, flatten(all_labels),)
            moving_objects = ZWhatEvaluator(cfg, title= "moving", **args).evaluate_z_what(z_whats, flatten(all_labels_moving),)
            relevant_objects = ZWhatEvaluator(cfg, title= "relevant", **args).evaluate_z_what(z_whats_relevant, flatten(all_labels_relevant),)

            # add bayes accuracy metric for relevant objects based on Kalman filter
            z_encs_relevant = [[enc[rel_idx] for enc, rel_idx in zip(enc_seq, rel_seq)] for enc_seq, rel_seq in zip(z_encs, all_labels_relevant_idx)]
            bayes_accuracy = classify_encodings(cfg, z_encs_relevant, all_labels_relevant)
            relevant_objects[2]['bayes_accuracy'] = bayes_accuracy

            if cfg.save_relevant_objects:
                # Save relevant objects data to pickle file
                with open(f'{self.relevant_object_hover_path}/relevant_objects_{global_step:06}.pkl', 'wb') as output_file:
                    relevant_objects_data = {
                        'z_what': z_whats_relevant,
                        'labels': all_labels_relevant,
                        'image_refs': [image_refs[idx] for idx, yes in enumerate(all_labels_relevant_idx) if yes]
                    }
                    pickle.dump(relevant_objects_data, output_file, pickle.DEFAULT_PROTOCOL)
        else:
            print("Can't evaluate z_what, because no objects were found")
            return None

        return {'all': all_objects, 'moving': moving_objects, 'relevant': relevant_objects}

    def save_relevant_objects(self, global_step, image_refs, batch_size, img_path, i, z_where, z_pres):
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

    