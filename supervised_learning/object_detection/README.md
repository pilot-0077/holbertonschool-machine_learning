# Object Detection

This project introduces the core concepts and practical components used in
modern object detection systems with YOLOv3 and OpenCV.

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

## Tasks

### 0. Initialize Yolo

File: `0-yolo.py`

Initializes a YOLOv3 detector by loading a Keras Darknet model, class names,
confidence thresholds, and anchor boxes.

### 1. Process Outputs

File: `1-yolo.py`

Processes raw YOLO outputs into bounding boxes relative to the original image,
box confidence scores, and class probability scores.

### 2. Filter Boxes

File: `2-yolo.py`

Computes class scores for each predicted box and removes detections whose best
class score is below the configured class threshold.

### 3. Non-max Suppression

File: `3-yolo.py`

Applies class-wise non-max suppression using Intersection over Union to remove
duplicate overlapping detections while preserving the highest-scoring boxes.

### 4. Load Images

File: `4-yolo.py`

Loads images from a directory with OpenCV and returns both the image arrays and
their corresponding file paths.

### 5. Preprocess Images

File: `5-yolo.py`

Resizes images to the model input dimensions, rescales pixel values to the
range `[0, 1]`, and records each image's original height and width.

### 6. Show Boxes

File: `6-yolo.py`

Draws predicted bounding boxes, class names, and confidence scores on an image.
The displayed detection can also be saved to the `detections` directory.

### 7. Predict

File: `7-yolo.py`

Runs the complete detection pipeline on every image in a directory: loading,
preprocessing, model inference, output processing, filtering, non-max
suppression, visualization, and returning the final predictions.

## Repository

- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/object_detection`
