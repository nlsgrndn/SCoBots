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
from model.space.space import Space
from model.space.inference_space import WrappedSPACEforInference
from utils.checkpointer import Checkpointer
import os.path as osp
from engine.utils import get_config_v2
import pandas as pd

# get current directory as absolute path
space_and_moc_base_path = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
models_path = osp.join(space_and_moc_base_path, "final_detect_models")


def load_classifier(game_name):
    classifier_path = osp.join(models_path, game_name, "classifier", "z_what-classifier_relevant.joblib.pkl")
    classifier = joblib.load(classifier_path)
    centroid_labels_path = osp.join(models_path, game_name, "classifier", "z_what-classifier_relevant_centroid_labels.csv")
    centroid_labels = pd.read_csv(centroid_labels_path, header=None, index_col=0)
    centroid_labels_dict = centroid_labels.iloc[:,0].to_dict()
    return classifier, centroid_labels_dict
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
    classifier, classifier_id_dict = load_classifier(game_name)
    wrapped_space = load_space_for_inference(game_name)
    space_detector = SPACE(
        game_name=game_name,
        classifier=classifier,
        classifier_id_dict=classifier_id_dict,
        wrapped_space = wrapped_space,
        object_names= {k:str(k) for k in range(4)}, #TODO fix (probably use OCAtari MAX_NB_OBJECT stuff)
        confidence_threshold=0.4,
        nms_threshold=0.2,
        draw_bboxes=True,
    )
    return space_detector