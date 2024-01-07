import cv2 as cv
from motrackers.detectors import SPOCDummy
from motrackers.tracking.centroid_kf_tracker import CentroidKF_Tracker
from motrackers.tracking.centroid_tracker import CentroidTracker
from motrackers.utils import draw_tracks
import numpy as np
import pandas as pd
import joblib
import os
from utils.bbox_matching import match_bounding_boxes_z_what, match_bounding_boxes_v2
from collections import Counter
from dataset.z_what import Atari_Z_What
from torch.utils.data import DataLoader
from engine.utils import get_config
# [4.0, 6.0, 2.0, 3.0]
skiing_labels_to_id = {0: 4, 1: 6, 2: 2, 3: 3}

def main(model, tracker, dataloader, classifier):
    os.makedirs('results_kalman', exist_ok=True)

    kf_pred_label_collector = []
    kf_gt_label_collector = []
    raw_pred_label_collector = []
    raw_gt_label_collector = []
    for i, data in enumerate(dataloader):
        if i<=0: # skip first batch
            continue
        
        # T signifies that the data is in time format (batch_size, seq_len, ...)
        pred_boxes_T = data["pred_boxes"]
        z_whats_T = data["z_whats_pres_s"]
        gt_labels_for_pred_boxes_T = data["gt_labels_for_pred_boxes"]
        gt_bbs_and_labels_T = data["gt_bbs_and_labels"]
        imgs_T = data["imgs"]
        pred_boxes_T, z_whats_T, gt_labels_for_pred_boxes_T, gt_bbs_and_labels_T, imgs_T = squeeze_all(pred_boxes_T, z_whats_T, gt_labels_for_pred_boxes_T, gt_bbs_and_labels_T, imgs_T)
        # change axis (0,1,2,3) to (0, 2, 3, 1)
        imgs_T = imgs_T.numpy().transpose(0, 2, 3, 1)

        
        for j, (pred_boxes, z_whats, gt_labels_for_pred_boxes, gt_bbs_and_labels, img)  in enumerate(zip(pred_boxes_T, z_whats_T, gt_labels_for_pred_boxes_T, gt_bbs_and_labels_T, imgs_T)):
        
            bboxes, confidences, class_ids, probabilities = model.detect(pred_boxes, z_whats)
            mask = filter_bboxes_close_to_border(bboxes)
            bboxes = bboxes[mask]
            confidences = confidences[mask]
            class_ids = class_ids[mask]
            probabilities = probabilities[mask]
            z_whats = z_whats[mask]


            # get pred labels for raw predictions and store

            raw_pred_labels = np.array([skiing_labels_to_id[int(label)] for label in class_ids])
            raw_pred_label_collector.extend(raw_pred_labels)

            # get gt labels for kf predictions and store
            raw_pred_bbox = bboxes.copy().astype(float)
            raw_pred_bbox = inverse_transform_bbox_format(raw_pred_bbox)
            raw_pred_bbox[:, :4] = raw_pred_bbox[:, :4] / 128.0
            raw_gt_labels = match_bounding_boxes_v2(gt_bbs_and_labels, raw_pred_bbox)
            raw_gt_label_collector.extend(raw_gt_labels)

            tracks = tracker.update(bboxes, confidences, class_ids, probabilities, z_whats, classifier)
            image = img.copy() * 255
            #updated_image = model.draw_bboxes(image, bboxes , confidences, class_ids)
            tracks_bbox = np.array([np.stack(track) for track in tracks])
            updated_image = model.draw_bboxes(image, (tracks_bbox[:, :4]).astype(int) , tracks_bbox[:, 5], tracks_bbox[:, 4])
            updated_image = draw_tracks(updated_image, tracks)

            # get pred labels for kf predictions
            tracks_bbox[:, 4] = np.array([skiing_labels_to_id[int(label)] for label in tracks_bbox[:, 4]])
            pred_labels = tracks_bbox[:, 4]
            kf_pred_label_collector.extend(pred_labels)

            # get gt labels for kf predictions and store
            tracks_bbox = inverse_transform_bbox_format(tracks_bbox)
            tracks_bbox[:, :4] = tracks_bbox[:, :4] / 128
            gt_labels = match_bounding_boxes_v2(gt_bbs_and_labels, tracks_bbox)
            kf_gt_label_collector.extend(gt_labels)

            # save image with cv
            cv.imwrite('results_kalman/' + str(j) + '.jpg', updated_image)


            #print_accuracy(gt_labels, pred_labels, dataset_name="KF")
            #print_accuracy(raw_gt_labels, raw_pred_labels, dataset_name="raw")



        # for all tracks print counter of self.prob_history_classification and self.class_id_history
        tracks = list(tracker.tracks.values())
        for track in tracks:
            print(f"Track {track.id}: {Counter(track.prob_history_classification)}, {Counter(track.class_id_history)}")

        break

    print_accuracy(kf_gt_label_collector, kf_pred_label_collector, dataset_name="KF")
    print_accuracy(raw_gt_label_collector, raw_pred_label_collector, dataset_name="raw")


def print_accuracy(gt_labels, pred_labels, dataset_name=""):
    gt_labels = np.array(gt_labels)
    pred_labels = np.array(pred_labels)
    accuracy = np.sum(gt_labels == pred_labels) / len(gt_labels)
    print(f"Accuracy {dataset_name}: {accuracy}")

def filter_bboxes_close_to_border(bboxes, border_x_threshold = 20):
    """
    Filter bboxes that are close to the border of the image.

    return: numpy mask of bboxes that are not close to the border
    """
    #import ipdb ; ipdb.set_trace()
    mask = np.ones(len(bboxes), dtype=bool)
    image_width, image_height = 128, 128
    for i, bbox in enumerate(bboxes):
        x_min, y_min, width, height = bbox
        center_x = x_min + width / 2
        center_y = y_min + height / 2
        if center_x < border_x_threshold or center_x > image_width - border_x_threshold:
            mask[i] = False
    return mask




    

def squeeze_all(*args):
    results = []
    for arg in args:
        if isinstance(arg, list):
            for i, a in enumerate(arg):
                arg[i] = a.squeeze(0)
            results.append(arg)
        else:
            arg = arg.squeeze(0)
            results.append(arg)
    return results


def inverse_transform_bbox_format( bboxes):
    """
    Transform from (xmin, ymin, width, height) to (y_min, y_max, x_min, x_max) format.
    """
    new_format_bboxes = np.array(bboxes)
    new_format_bboxes[:, 0] = bboxes[:, 1]
    new_format_bboxes[:, 2] = bboxes[:, 0] 
    new_format_bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3]
    new_format_bboxes[:, 3] = bboxes[:, 0] + bboxes[:, 2]
    return new_format_bboxes


def get_data_for_kalman(cfg):
    FILTERED_PREDICTED_BBS = True
    filter_str = "filtered" if FILTERED_PREDICTED_BBS else "unfiltered" #TODO replace

    game = cfg.gamelist[0].split("-")[0].lower() # TODO configure somewhere else
    boxes_subset = "all" # TODO configure somewhere else
    dataset_mode = "test" # TODO configure somewhere else
    atari_z_what_dataset = Atari_Z_What(cfg, dataset_mode, boxes_subset, return_keys = ["z_whats_pres_s", "gt_labels_for_pred_boxes", "imgs", "pred_boxes", "gt_bbs_and_labels"])
    atari_z_what_dataloader = DataLoader(atari_z_what_dataset, batch_size=1, shuffle=False, num_workers=0) #batch_size must be 1
    z_classifier_path = f"../output/logs/{game}/model_000005001/z_what-classifier_{filter_str}.joblib.pkl"
    z_what_classifier = joblib.load(z_classifier_path)
    
    return z_what_classifier, atari_z_what_dataloader

def inverse_z_where_to_bb_format(y_min, y_max, x_min, x_max):
    width = x_max - x_min
    height = y_max - y_min
    center_x = x_min + width / 2
    center_y = y_min + height / 2
    center_x = center_x * 2.0 - 1.0
    center_y = center_y * 2.0 - 1.0
    return width, height, center_x, center_y

def execute(cfg, args = None):
    if args:
        if args.tracker == 'CentroidTracker':
            tracker = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')
        elif args.tracker == 'CentroidKF_Tracker':
            tracker = CentroidKF_Tracker(max_lost=1,
                                     centroid_distance_threshold= 10.0,
                                     process_noise_scale= 1,
                                     measurement_noise_scale= 1,
                                     tracker_output_format='my_format') # max_lost was 0
        else:
            raise NotImplementedError
    else: # default
        tracker = CentroidKF_Tracker(max_lost=2,
                         centroid_distance_threshold= 5.0,
                         process_noise_scale= 1,
                         measurement_noise_scale= 1,
                         tracker_output_format='my_format') # max_lost was 0

    classifier, dataloader= get_data_for_kalman(cfg)
    model = SPOCDummy(
        classifier=classifier,
        object_names= {k:str(k) for k in range(4)},
        confidence_threshold=0.4,
        nms_threshold=0.2,
        draw_bboxes=True,
    )
    
    main(model, tracker, dataloader, classifier)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Object detections in input video using TensorFlow model of MobileNetSSD.')

    parser.add_argument(
        '--gpu', type=bool, default=False,
        help='Flag to use gpu to run the deep learning model. Default is `False`'
    )

    parser.add_argument(
        '--tracker', type=str, default='CentroidKF_Tracker',
        help="Tracker used to track objects. Options include ['CentroidTracker', 'CentroidKF_Tracker']")
 
    args = parser.parse_args()
    
    cfg = None
    print("Probably not working, because of missing cfg. Try to execute via main.py")
    execute(cfg, args)

    