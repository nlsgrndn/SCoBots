import numpy as np
from motrackers.kalman_tracker import KalmanFilter2DConstantAcc, KalmanFilter2DConstantVel
from collections import Counter 

class Track:
    """
    Track containing attributes to track various objects.

    Args:
        frame_id (int): Camera frame id.
        track_id (int): Track Id
        bbox (numpy.ndarray): Bounding box pixel coordinates as (xmin, ymin, width, height) of the track.
        detection_confidence (float): Detection confidence of the object (probability).
        class_id (str or int): Class label id.
        lost (int): Number of times the object or track was not tracked by tracker in consecutive frames.
        iou_score (float): Intersection over union score.
        data_output_format (str): Output format for data in tracker.
            Options include ``['mot_challenge', 'visdrone_challenge']``. Default is ``mot_challenge``.
        kwargs (dict): Additional key word arguments.

    """

    count = 0

    def __init__(self, track_id, frame_id, bbox, detection_confidence, class_id=None, lost=0, iou_score=0.,
                data_output_format = 'mot_challenge', **kwargs):
        
        self.id = track_id

        self.detection_confidence_max = 0.
        self.lost = 0
        self.age = 0
        self.class_id_history = []

        self.update(frame_id, bbox, detection_confidence, class_id=class_id, lost=lost, iou_score=iou_score, **kwargs)

        if data_output_format == 'mot_challenge':
            self.output = self.get_mot_challenge_format
        elif data_output_format == 'my_format':
            self.output = self.get_my_format
        else:
            raise NotImplementedError

    def update(self, frame_id, bbox, detection_confidence, class_id=None, lost=0, iou_score=0., **kwargs):
        """
        Update the track.

        Args:
            frame_id (int): Camera frame id.
            bbox (numpy.ndarray): Bounding box pixel coordinates as (xmin, ymin, width, height) of the track.
            detection_confidence (float): Detection confidence of the object (probability).
            class_id (int or str): Class label id.
            lost (int): Number of times the object or track was not tracked by tracker in consecutive frames.
            iou_score (float): Intersection over union score.
            kwargs (dict): Additional key word arguments.
        """
        self.class_id = class_id
        self.bbox = np.array(bbox)
        self.detection_confidence = detection_confidence
        self.frame_id = frame_id
        self.iou_score = iou_score

        if lost == 0:
            self.lost = 0
        else:
            self.lost += lost

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.detection_confidence_max = max(self.detection_confidence_max, detection_confidence)

        self.age += 1

    #@property
    #def centroid(self):
    #    """
    #    Return the centroid of the bounding box.
#
    #    Returns:
    #        numpy.ndarray: Centroid (x, y) of bounding box.
#
    #    """
    #    return np.array((self.bbox[0]+0.5*self.bbox[2], self.bbox[1]+0.5*self.bbox[3]))
    
    @staticmethod
    def centroid(bbox):
        """
        Return the centroid of the bounding box.

        Returns:
            numpy.ndarray: Centroid (x, y) of bounding box.

        """
        return np.array((bbox[0]+0.5*bbox[2], bbox[1]+0.5*bbox[3]))

    def get_mot_challenge_format(self):
        """
        Get the tracker data in MOT challenge format as a tuple of elements containing
        `(frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z)`

        References:
            - Website : https://motchallenge.net/

        Returns:
            tuple: Tuple of 10 elements representing `(frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z)`.

        """
        mot_tuple = (
            self.frame_id, self.id, self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], self.detection_confidence,
            -1, -1, -1
        )
        return mot_tuple
    
    def get_my_format(self):
        """
        Get the tracker data

        Returns:
            tuple: Tuple of 10 elements representing '(bb_left, bb_top, bb_width, bb_height, class_id, conf)'

        """
        raise NotImplementedError("only implemented for KFTrackCentroid")

    def predict(self):
        """
        Implement to prediction the next estimate of track.
        """
        raise NotImplemented

class KFTrackCentroid(Track):
    """
    Track based on Kalman filter used for Centroid Tracking of bounding box in MOT.

    Args:
        track_id (int): Track Id
        frame_id (int): Camera frame id.
        bbox (numpy.ndarray): Bounding box pixel coordinates as (xmin, ymin, width, height) of the track.
        detection_confidence (float): Detection confidence of the object (probability).
        class_id (str or int): Class label id.
        lost (int): Number of times the object or track was not tracked by tracker in consecutive frames.
        iou_score (float): Intersection over union score.
        data_output_format (str): Output format for data in tracker.
            Options ``['mot_challenge', 'visdrone_challenge']``. Default is ``mot_challenge``.
        process_noise_scale (float): Process noise covariance scale or covariance magnitude as scalar value.
        measurement_noise_scale (float): Measurement noise covariance scale or covariance magnitude as scalar value.
        kwargs (dict): Additional key word arguments.
    """
    def __init__(self, track_id, frame_id, bbox, detection_confidence, class_id=None, lost=0, iou_score=0.,
                 data_output_format='mot_challenge', process_noise_scale=1.0, measurement_noise_scale=1.0, **kwargs):
        c = np.array((bbox[0]+0.5*bbox[2], bbox[1]+0.5*bbox[3]))
        self.kf = KalmanFilter2DConstantAcc(c, process_noise_scale=process_noise_scale, measurement_noise_scale=measurement_noise_scale)
        self.prob_history = []
        self.prob_history_classification = []
        self.class_id_history = []
        super().__init__(track_id, frame_id, bbox, detection_confidence, class_id=class_id, lost=lost,
                         iou_score=iou_score, data_output_format=data_output_format, **kwargs)

    def predict(self):
        """
        Predicts the next estimate of the bounding box of the track.

        Returns:
            numpy.ndarray: Bounding box pixel coordinates as (xmin, ymin, width, height) of the track.

        """
        self.kf.predict()
        xmid, ymid = self.kf.get_xy()
        w, h = self.bbox[2], self.bbox[3]
        xmin = xmid - 0.5*w
        ymin = ymid - 0.5*h
        return np.array([xmin, ymin, w, h]).astype(int)

    def update(self, frame_id, bbox, detection_confidence, class_id=None, lost=0, iou_score=0., **kwargs):
        measured_centroid = self.centroid(bbox)
        self.kf.update(measured_centroid)
        # update bbox coordinates
        xmid, ymid = self.kf.get_xy()
        w, h = bbox[2], bbox[3]
        xmin = xmid - 0.5*w
        ymin = ymid - 0.5*h
        bbox = np.array([xmin, ymin, w, h]).astype(int)
        # check if probability_for_track is in kwargs
        if 'probabilities_for_track' in kwargs:
            self.prob_history.append(kwargs['probabilities_for_track'])
            del kwargs['probabilities_for_track']

        self.class_id_history.append(class_id)
        class_id_with_max_prob_history = np.array(self.prob_history).sum(axis=0).argmax()
        self.prob_history_classification.append(class_id_with_max_prob_history)
        super().update(
            frame_id, bbox, detection_confidence, class_id=class_id, lost=lost, iou_score=iou_score, **kwargs)
        #print(self.centroid)
    
    def get_my_format(self):
        """
        Get the tracker data

        Returns:
            tuple: Tuple of 10 elements representing '(bb_left, bb_top, bb_width, bb_height, class_id, conf)'

        """

        #if len(self.prob_history) > 10:
        #    import ipdb; ipdb.set_trace()
        mot_tuple = (
            self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], self.prob_history_classification[-1], self.detection_confidence,
        )
        return mot_tuple