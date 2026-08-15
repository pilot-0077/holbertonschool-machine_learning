#!/usr/bin/env python3
"""Initializes a YOLOv3 object detector."""

from tensorflow import keras as K


class Yolo:
    """Uses the YOLOv3 algorithm to perform object detection."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize a YOLOv3 detector.

        Args:
            model_path: Path to the stored Darknet Keras model.
            classes_path: Path to the file containing class names.
            class_t: Box score threshold for initial filtering.
            nms_t: IOU threshold for non-max suppression.
            anchors: NumPy array containing the anchor boxes.
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r', encoding='utf-8') as classes_file:
            self.class_names = [
                class_name.strip()
                for class_name in classes_file.readlines()
            ]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
