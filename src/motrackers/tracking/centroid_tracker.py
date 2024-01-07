import numpy as np
from scipy.spatial import distance
from motrackers.tracking.tracker import Tracker
from motrackers.utils.misc import get_centroid

class CentroidTracker(Tracker):
    """
    Greedy Tracker with tracking based on ``centroid`` location of the bounding box of the object.

    Args:
        max_lost (int): Maximum number of consecutive frames object was not detected.
        tracker_output_format (str): Output format of the tracker.
    """

    def __init__(self, max_lost=5, tracker_output_format='mot_challenge'):
        super().__init__(max_lost, tracker_output_format)

    def update(self, bboxes, detection_scores, class_ids):
        """
        Update the tracker based on the new bounding boxes.

        Args:
            bboxes (numpy.ndarray or list): List of bounding boxes detected in the current frame. Each element of the list represent
                coordinates of bounding box as tuple `(top-left-x, top-left-y, width, height)`.
            detection_scores(numpy.ndarray or list): List of detection scores (probability) of each detected object.
            class_ids (numpy.ndarray or list): List of class_ids (int) corresponding to labels of the detected object. Default is `None`.

        Returns:
            list: List of tracks being currently tracked by the tracker. Each track is represented by the tuple with elements `(frame_id, track_id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z)`.
        """

        self.frame_count += 1

        if len(bboxes) == 0:
            lost_ids = list(self.tracks.keys())

            for track_id in lost_ids:
                self.tracks[track_id].lost += 1
                if self.tracks[track_id].lost > self.max_lost:
                    self._remove_track(track_id)

            outputs = self._get_tracks(self.tracks)
            return outputs

        detections = CentroidTracker.preprocess_input(bboxes, class_ids, detection_scores)

        track_ids = list(self.tracks.keys())

        updated_tracks, updated_detections = [], []

        if len(track_ids):
            track_centroids = np.array([self.tracks[tid].centroid(self.tracks[tid].bbox) for tid in track_ids])
            detection_centroids = get_centroid(np.asarray(bboxes))

            centroid_distances = distance.cdist(track_centroids, detection_centroids)

            track_indices = np.amin(centroid_distances, axis=1).argsort()

            for idx in track_indices:
                track_id = track_ids[idx]

                remaining_detections = [
                    (i, d) for (i, d) in enumerate(centroid_distances[idx, :]) if i not in updated_detections]

                if len(remaining_detections):
                    detection_idx, detection_distance = min(remaining_detections, key=lambda x: x[1])
                    bbox, class_id, confidence = detections[detection_idx]
                    self._update_track(track_id, self.frame_count, bbox, confidence, class_id=class_id)
                    updated_detections.append(detection_idx)
                    updated_tracks.append(track_id)

                if len(updated_tracks) == 0 or track_id is not updated_tracks[-1]:
                    self.tracks[track_id].lost += 1
                    if self.tracks[track_id].lost > self.max_lost:
                        self._remove_track(track_id)

        for i, (bbox, class_id, confidence) in enumerate(detections):
            if i not in updated_detections:
                self._add_track(self.frame_count, bbox, confidence, class_id=class_id)

        outputs = self._get_tracks(self.tracks)
        return outputs