import numpy as np
import cv2 as cv
from motrackers.detectors.detector import Detector
from motrackers.utils.misc import load_labelsjson


# with temperature
def softmax(x, temperature=1):
    return np.exp(x / temperature) / np.sum(np.exp(x / temperature), axis=1, keepdims=True)
class SPOC(Detector):
    """
    Abstract class for detector.

    Args:
        classifier (Classifier): Classifier to classify the objects.
        object_names (dict): Dictionary containing (key, value) as (class_id, class_name) for object detector.
        confidence_threshold (float): Confidence threshold for object detection.
        nms_threshold (float): Threshold for non-maximal suppression.
        draw_bboxes (bool): If true, draw bounding boxes on the image is possible.
    """

    def __init__(self, classifier, object_names, confidence_threshold, nms_threshold, draw_bboxes=True):
        self.classifier = classifier
        super().__init__(object_names, confidence_threshold, nms_threshold, draw_bboxes)

    def forward(self, image):
        """
        Forward pass for the detector with input image.

        Args:
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: detections
        """
        raise NotImplemented

    def detect(self, pred_boxes, z_whats,):
        """
        Detect objects in the input image.

        Args:
            image (numpy.ndarray): Input image.

        Returns:
            tuple: Tuple containing the following elements:
                - bboxes (numpy.ndarray): Bounding boxes with shape (n, 4) containing detected objects with each row as `(xmin, ymin, width, height)`.
                - confidences (numpy.ndarray): Confidence or detection probabilities if the detected objects with shape (n,).
                - class_ids (numpy.ndarray): Class_ids or label_ids of detected objects with shape (n, 4)

        """
        predbboxs = pred_boxes[:, :4]
        bboxes = self.transform_bbox_format(predbboxs)
        bboxes = np.array(bboxes * 128).astype(np.int32)
        z_whats = z_whats
        #class_ids = self.classifier.predict(z_whats)
        distances = self.classifier.transform(z_whats)
        probabilities = softmax(-distances)
        class_ids = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)
        
        return bboxes, confidences, class_ids, probabilities


    def transform_bbox_format(self, bboxes):
        """
        Transform from (y_min, y_max, x_min, x_max) to (xmin, ymin, width, height) format.
        """
        new_format_bboxes = np.array(bboxes)
        new_format_bboxes[:, 0] = bboxes[:, 2]
        new_format_bboxes[:, 1] = bboxes[:, 0]
        new_format_bboxes[:, 2] = bboxes[:, 3] - bboxes[:, 2]
        new_format_bboxes[:, 3] = bboxes[:, 1] - bboxes[:, 0]
        return new_format_bboxes
    
