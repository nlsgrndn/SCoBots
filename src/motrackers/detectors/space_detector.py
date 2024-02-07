import numpy as np
import cv2 as cv
from motrackers.detectors.detector import Detector
from motrackers.utils.misc import load_labelsjson
from model.space.postprocess_latent_variables import latent_to_boxes_and_z_whats
import torch
from dataset.atari_labels import filter_relevant_boxes_masks

# with temperature
def softmax(x, temperature=1):
    return np.exp(x / temperature) / np.sum(np.exp(x / temperature), axis=1, keepdims=True)
class SPACEDummy(Detector):
    """
    Class for dummy space detector.

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
    

class SPACE(Detector):
    """
    class for detector based on SPACE

    Args:
        classifier (Classifier): Classifier to classify the objects.
        object_names (dict): Dictionary containing (key, value) as (class_id, class_name) for object detector.
        confidence_threshold (float): Confidence threshold for object detection.
        nms_threshold (float): Threshold for non-maximal suppression.
        draw_bboxes (bool): If true, draw bounding boxes on the image is possible.
    """

    def __init__(self, game_name, classifier, wrapped_space, object_names, confidence_threshold, nms_threshold, draw_bboxes=True):
        self.game_name = game_name
        self.classifier = classifier
        self.wrapped_space = wrapped_space
        super().__init__(object_names, confidence_threshold, nms_threshold, draw_bboxes)

    def forward(self, image):
        """
        Forward pass for the detector with input image.

        Args:
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: detections
        """
        self.wrapped_space.eval()
        with torch.no_grad():
            latent_logs_dict_wrapped = self.wrapped_space.forward(image)
        return latent_logs_dict_wrapped

    def detect(self, image):
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

        latent_logs_dict = self.forward(image)
        predbboxs, z_whats = latent_to_boxes_and_z_whats(latent_logs_dict)
        if len(z_whats) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        mask = filter_relevant_boxes_masks(self.game_name, predbboxs, None)[0]
        if not torch.any(mask):
            return np.array([]), np.array([]), np.array([]), np.array([])
        predbboxs = predbboxs[0][mask]
        z_whats = z_whats[mask]
        predbboxs = predbboxs.to("cpu").numpy()
        z_whats = z_whats.to("cpu").numpy()
        bboxes = self.transform_bbox_format(predbboxs)
        bboxes = np.array(bboxes * 128)
        #bboxes = bboxes.astype(np.int32)

        #distances = self.classifier.transform(z_whats) #TODO: generalize this to other classifiers
        #probabilities = softmax(-distances)
#
        #class_ids = np.argmax(probabilities, axis=1)
        #class_ids = np.array([self.classifier_id_dict[class_id] for class_id in class_ids])
        #confidences = np.max(probabilities, axis=1)

        class_ids = self.classifier.predict(z_whats)
        confidences = np.zeros(len(class_ids))
        #class_ids = np.array([self.classifier_id_dict[class_id] for class_id in class_ids])

        return bboxes, confidences, class_ids

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
