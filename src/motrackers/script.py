import cv2 as cv
from motrackers.detectors import SPOC
from motrackers import CentroidTracker, CentroidKF_Tracker
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


#def main(model, tracker, data, actual_bbs_and_labels):
#    i = 0
#    # make results dir if it doesn't exist
#    os.makedirs('results_kalman', exist_ok=True)
#
#    pred_label_collector = []
#    gt_label_collector = []
#    while i < len(data):
#        #if i >= 10:
#        #    break
#        
#        predbboxs_predlabels_gtlabels_z_whats = data[i]
#        curr_actual_bbs_and_labels = actual_bbs_and_labels[i]
#        mask = np.isin(curr_actual_bbs_and_labels[:, 4], [2,3,4,6])
#        curr_actual_bbs_and_labels = curr_actual_bbs_and_labels[mask]
#        labels = predbboxs_predlabels_gtlabels_z_whats[:, 5]
#        image = cv.imread(f"../consecutive_atari_data/Skiing-v0/space_like/test/00000_{i}.png")
#        image = cv.resize(image, (128, 128))
#        
#        bboxes, confidences, class_ids, probabilities = model.detect(predbboxs_predlabels_gtlabels_z_whats)
#        bboxes_pred_kf = tracker.predict()
#        #print_tracks(tracker)
#        #import ipdb; ipdb.set_trace()
#        #predicted_kf_image = model.draw_bboxes(image.copy(), bboxes_pred_kf , confidences, class_ids)
#        #cv.imwrite('results_kalman/' + str(i) + '_predicted_kf.jpg', predicted_kf_image)
#
#        tracks = tracker.update(bboxes, confidences, class_ids, bboxes_pred_kf, probabilities)
#        #print_tracks(tracker)
#        #import ipdb; ipdb.set_trace()
#        #tracks_bbox = np.array([np.stack((track[2], track[3], track[4], track[5])) for track in tracks])
#        #tracks_image = model.draw_bboxes(predicted_kf_image, tracks_bbox , confidences, class_ids)
#        #cv.imwrite('results_kalman/' + str(i) + 'predictedkf_vs_updated.jpg', tracks_image)
#
#        tracks_bbox = np.array([np.stack(track) for track in tracks])
#        tracks_bbox = inverse_transform_bbox_format(tracks_bbox)
#        tracks_bbox[:, :4] = tracks_bbox[:, :4] / 128
#        tracks_bbox[:, 4] = np.array([skiing_labels_to_id[int(label)] for label in tracks_bbox[:, 4]])
#        # match tracks with actual bbs
#        gt_labels = match_bounding_boxes_v2(curr_actual_bbs_and_labels, tracks_bbox)
#        pred_labels = tracks_bbox[:, 4]
#        gt_label_collector.extend(gt_labels)
#        pred_label_collector.extend(pred_labels)
#        updated_image = model.draw_bboxes(image.copy(), bboxes , confidences, class_ids)
#
#        updated_image = draw_tracks(updated_image, tracks)
#
#        # save image with cv
#        cv.imwrite('results_kalman/' + str(i) + '.jpg', updated_image)
#        i += 1
#
#        #import ipdb; ipdb.set_trace()
#    # calculate accuracy
#        
#    # for all tracks print counter of self.prob_history_classification and self.class_id_history
#    tracks = list(tracker.tracks.values())
#    for track in tracks:
#        print(f"Track {track.id}: {Counter(track.prob_history_classification)}, {Counter(track.class_id_history)}")
#
#    gt_label_collector = np.array(gt_label_collector)
#    pred_label_collector = np.array(pred_label_collector)
#    accuracy = np.sum(gt_label_collector == pred_label_collector) / len(gt_label_collector)
#    print(f"Accuracy: {accuracy}")

def main(model, tracker, dataloader):
    os.makedirs('results_kalman', exist_ok=True)

    pred_label_collector = []
    gt_label_collector = []
    for i, data in enumerate(dataloader):
        if i==0: # skip first batch
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

        
        for pred_boxes, z_whats, gt_labels_for_pred_boxes, gt_bbs_and_labels, img in zip(pred_boxes_T, z_whats_T, gt_labels_for_pred_boxes_T, gt_bbs_and_labels_T, imgs_T):
        
            bboxes, confidences, class_ids, probabilities = model.detect(pred_boxes, z_whats)
            bboxes_pred_kf = tracker.predict()
            tracks = tracker.update(bboxes, confidences, class_ids, bboxes_pred_kf, probabilities)
            image = img.copy() * 255
            #updated_image = model.draw_bboxes(image, bboxes , confidences, class_ids)
            tracks_bbox = np.array([np.stack(track) for track in tracks])
            updated_image = model.draw_bboxes(image, (tracks_bbox[:, :4]).astype(int) , tracks_bbox[:, 5], tracks_bbox[:, 4])
            updated_image = draw_tracks(updated_image, tracks)


            
            tracks_bbox = inverse_transform_bbox_format(tracks_bbox)
            tracks_bbox[:, :4] = tracks_bbox[:, :4] / 128
            tracks_bbox[:, 4] = np.array([skiing_labels_to_id[int(label)] for label in tracks_bbox[:, 4]])
            # match tracks with gt bbs
            gt_labels = match_bounding_boxes_v2(gt_bbs_and_labels, tracks_bbox)
            pred_labels = tracks_bbox[:, 4]
            gt_label_collector.extend(gt_labels)
            pred_label_collector.extend(pred_labels)
            


            # save image with cv
            cv.imwrite('results_kalman/' + str(i) + '.jpg', updated_image)

            gt_labels = np.array(gt_labels)
            pred_labels = np.array(pred_labels)
            accuracy = np.sum(gt_labels == pred_labels) / len(gt_labels)
            print(f"Accuracy: {accuracy}")
            # print track infos
            print(tracker._get_tracks(tracker.tracks))
            #import ipdb; ipdb.set_trace()

            i += 1

        # for all tracks print counter of self.prob_history_classification and self.class_id_history
        tracks = list(tracker.tracks.values())
        for track in tracks:
            print(f"Track {track.id}: {Counter(track.prob_history_classification)}, {Counter(track.class_id_history)}")

        break

    gt_label_collector = np.array(gt_label_collector)
    pred_label_collector = np.array(pred_label_collector)
    accuracy = np.sum(gt_label_collector == pred_label_collector) / len(gt_label_collector)
    print(f"Accuracy: {accuracy}")

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


#def get_data_for_kalman():
#    FILTERED_PREDICTED_BBS = True
#    filter_str = "filtered" if FILTERED_PREDICTED_BBS else "unfiltered"
#
#    game = "skiing"
#
#    # pong [2.0, 1.0, 4.0]
#    # skiing [4.0, 2.0, 6.0, 3.0]
#    #k_means_centroid_ids_to_labels = {"skiing": {0: 4.0, 1: 2.0, 2: 6.0, 3: 3.0},
#    #                                    "boxing": {0: 2.0, 1: 6.0, 2: 4.0, 3: 3.0}, #TODO: change to correct labels
#    #                                    "pong": {0: 2.0, 1: 1.0, 2: 4.0,}}
#
#
#    z_classifier_path = f"../output/logs/{game}/model_000005001/z_what-classifier_{filter_str}.joblib.pkl"
#    z_what_classifier = joblib.load(z_classifier_path)
#
#    folder = "test"
#    data = np.load(f"labeled/{game}/predbboxs_predlabels_gtlabels_z_whats_{folder}_{filter_str}.npz")
#    actual_bbs_and_labels = np.load(f"labeled/{game}/actual_bbs_and_label_{folder}.npz")
#    #image_refs = pd.read_csv(f"labeled/{game}/image_refs_{folder}_{filter_str}.csv", header=None).iloc[:, 0].values
#    observations_tmp = []
#    actual_bbs_and_label_collector = []
#
#    for i, key in enumerate(list(data.keys())):
#        predbboxs_predlabels_gtlabels_z_whats = data["arr_" + str(i)]
#        actual_bbs_and_label_collector.append(actual_bbs_and_labels["arr_" + str(i)])
#        #predbboxs = predbboxs_predlabels_gtlabels_z_whats[:, 0:4]
#        #predbboxs = np.stack(inverse_z_where_to_bb_format(predbboxs[:, 0], predbboxs[:, 1], predbboxs[:, 2], predbboxs[:, 3]), axis=1)
#        #predbboxs_predlabels_gtlabels_z_whats[:, 0:4] = predbboxs
#        observations_tmp.append(predbboxs_predlabels_gtlabels_z_whats)
#        if i>= 127: #TODO change i%128 == 127 to allow for multiple sequences (image_refs then also needs to be changed)
#            break 
#
#    
#    return z_what_classifier, observations_tmp, actual_bbs_and_label_collector

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
    model = SPOC(
        classifier=classifier,
        object_names= {k:str(k) for k in range(4)},
        confidence_threshold=0.4,
        nms_threshold=0.2,
        draw_bboxes=True,
    )
    
    main(model, tracker, dataloader)

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

    