"""
Multi-object Trackers in Python:
    - GitHub link: https://github.com/adipandas/multi-object-tracker
    - Author: Aditya M. Deshpande
    - Blog: http://adipandas.github.io/
"""


from motrackers.tracking.centroid_tracker import CentroidTracker
from motrackers.tracking.centroid_kf_tracker import CentroidKF_Tracker
from motrackers.detectors.space_detector import SPACE


import joblib
from  space_models.space.space import Space
from  space_models.space.inference_space import WrappedSPACEforInference
from space_and_moc_utils.checkpointer import Checkpointer
import os.path as osp
from engine.utils import get_config_v2
import pandas as pd

# get current directory as absolute path
space_and_moc_base_path = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
models_path = osp.join(space_and_moc_base_path, "scobots_spaceandmoc_detectors")


def load_classifier(game_name):
    classifier_file_name = "z_what-classifier_relevant_nn.joblib.pkl"
    classifier_path = osp.join(models_path, game_name, "classifier", classifier_file_name)
    classifier = joblib.load(classifier_path)
    return classifier

def load_space_for_inference(game_name):
    config_path = osp.join(space_and_moc_base_path, "src", "configs", f"my_atari_{game_name}_gpu.yaml") #TODO specify config path in a better way
    cfg = get_config_v2(config_path) # get config must be called because it updates e.g. arch.G which is set differently in different scobi.config
    device = "cuda"
    model_path = osp.join(models_path, game_name, "space_weights", "model_000005001.pth")
    model = Space()
    model = model.to(device) #TODO: get device from somewhere
    checkpointer = Checkpointer("dummy_path", max_num=4) #TODO replace dummy path
    checkpointer.load(model_path, model, None, None, device)
    model = WrappedSPACEforInference(model)
    return model

def load_space_detector(game_name):
    classifier = load_classifier(game_name)
    wrapped_space = load_space_for_inference(game_name)
    space_detector = SPACE(
        game_name=game_name,
        classifier=classifier,
        wrapped_space = wrapped_space,
        object_names= {k:str(k) for k in range(4)}, #TODO fix (probably use OCAtari MAX_NB_OBJECT stuff)
        confidence_threshold=0.4,
        nms_threshold=0.2,
        draw_bboxes=True,
    )
    return space_detector