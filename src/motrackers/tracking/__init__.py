from motrackers.tracking.centroid_tracker import CentroidTracker
from motrackers.tracking.centroid_kf_tracker import CentroidKF_Tracker

def get_tracker(name):
    if name == "CentroidTracker":
        return CentroidTracker(max_lost=0)
    elif name == "CentroidKF_Tracker":
        return CentroidKF_Tracker(max_lost=0)
    else:
        raise ValueError(f"Unknown tracker name: {name}")