# Object Detection

This project introduces the core concepts and practical components used in
modern object detection systems.

## Learning Objectives

By the end of this project, I should be able to explain:

- What OpenCV is and how it is used
- What object detection is
- How the Sliding Windows algorithm works
- What a single-shot detector is
- How the YOLO algorithm works
- What Intersection over Union (IoU) is and how it is calculated
- What non-max suppression is
- What anchor boxes are
- What mean Average Precision (mAP) is and how it is calculated

## Task 0 - Initialize Yolo

File: `0-yolo.py`

Implements the `Yolo` class constructor for a YOLOv3 object detector.

The constructor:

- Loads a stored Darknet Keras model
- Reads the model's class names from a text file
- Stores the class score threshold
- Stores the non-max suppression IoU threshold
- Stores the anchor boxes used by the model

The class exposes the following public attributes:

- `model`
- `class_names`
- `class_t`
- `nms_t`
- `anchors`

## Repository

- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/object_detection`
