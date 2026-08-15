#!/usr/bin/env python3
"""Filters YOLOv3 bounding boxes by class score."""

import numpy as np
from tensorflow import keras as K


class Yolo:
    """Uses the YOLOv3 algorithm to perform object detection."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize a YOLOv3 detector."""
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r', encoding='utf-8') as classes_file:
            self.class_names = [
                class_name.strip()
                for class_name in classes_file.readlines()
            ]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def sigmoid(x):
        """Apply the sigmoid function to x."""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """Process raw Darknet outputs into image-relative boxes."""
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size
        model_dim_1 = int(self.model.input_shape[1])
        model_dim_2 = int(self.model.input_shape[2])

        for output_index, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes = output.shape[:3]

            grid_x = np.arange(grid_width).reshape(1, grid_width, 1)
            grid_y = np.arange(grid_height).reshape(grid_height, 1, 1)

            box_x = (
                self.sigmoid(output[..., 0]) + grid_x
            ) / grid_width
            box_y = (
                self.sigmoid(output[..., 1]) + grid_y
            ) / grid_height

            anchors = self.anchors[output_index].astype(float)
            anchor_widths = anchors[:, 0].reshape(1, 1, anchor_boxes)
            anchor_heights = anchors[:, 1].reshape(1, 1, anchor_boxes)

            box_width = (
                np.exp(output[..., 2]) * anchor_widths / model_dim_1
            )
            box_height = (
                np.exp(output[..., 3]) * anchor_heights / model_dim_2
            )

            processed_boxes = np.empty_like(output[..., :4], dtype=float)
            processed_boxes[..., 0] = (
                box_x - box_width / 2
            ) * image_width
            processed_boxes[..., 1] = (
                box_y - box_height / 2
            ) * image_height
            processed_boxes[..., 2] = (
                box_x + box_width / 2
            ) * image_width
            processed_boxes[..., 3] = (
                box_y + box_height / 2
            ) * image_height

            boxes.append(processed_boxes)
            box_confidences.append(self.sigmoid(output[..., 4:5]))
            box_class_probs.append(self.sigmoid(output[..., 5:]))

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes whose highest class score is below class_t."""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, confidence, class_probs in zip(
                boxes, box_confidences, box_class_probs):
            scores = confidence * class_probs
            classes = np.argmax(scores, axis=-1)
            scores = np.max(scores, axis=-1)
            mask = scores >= self.class_t

            filtered_boxes.append(box[mask])
            box_classes.append(classes[mask])
            box_scores.append(scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores
