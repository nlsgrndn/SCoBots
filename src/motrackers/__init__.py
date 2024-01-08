"""
Multi-object Trackers in Python:
    - GitHub link: https://github.com/adipandas/multi-object-tracker
    - Author: Aditya M. Deshpande
    - Blog: http://adipandas.github.io/
"""


from motrackers.tracking.centroid_tracker import CentroidTracker
from motrackers.tracking.centroid_kf_tracker import CentroidKF_Tracker
from motrackers.detectors.spoc import SPOC


import joblib
from model.space.space import Space
from utils.checkpointer import Checkpointer
import os.path as osp
from engine.utils import get_config_v2
import pandas as pd

base_path = "final_detect_models"
def load_classifier(game_name):
    classifier_path = osp.join(base_path, game_name, "classifier", "z_what-classifier_filtered.joblib.pkl")
    classifier = joblib.load(classifier_path)
    centroid_labels_path = osp.join(base_path, game_name, "classifier", "z_what-classifier_filtered_centroid_labels.csv")
    centroid_labels = pd.read_csv(centroid_labels_path, header=None, index_col=0)
    centroid_labels_dict = centroid_labels.iloc[:,0].to_dict()
    return classifier, centroid_labels_dict
def load_space_model(game_name):
    config_path = f"src/configs/my_atari_{game_name}_gpu.yaml" #TODO specify config path in a better way
    cfg = get_config_v2(game_name, config_path) # get config must be called because it updates e.g. arch.G which is set differently in different scobi.config
    device = "cuda"
    model_path = osp.join(base_path, game_name, "space_weights", "model_000005001.pth")
    model = Space()
    model = model.to(device) #TODO: get device from somewhere
    checkpointer = Checkpointer("dummy_path", max_num=4)
    checkpointer.load(model_path, model, None, None, device)
    return model
def load_spoc_detector(game_name):
    classifier, classifier_id_dict = load_classifier(game_name)
    spoc = load_space_model(game_name)
    spoc_detector = SPOC(
        classifier=classifier,
        classifier_id_dict=classifier_id_dict,
        spoc = spoc,
        object_names= {k:str(k) for k in range(4)}, #TODO fix (probably use OCAtari MAX_NB_OBJECT stuff)
        confidence_threshold=0.4,
        nms_threshold=0.2,
        draw_bboxes=True,
    )
    return spoc_detector