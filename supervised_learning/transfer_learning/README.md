# Transfer Learning

This project covers transfer learning and fine-tuning using pretrained
convolutional neural networks from Keras Applications.

## Learning Objectives

- What transfer learning is
- What fine-tuning is
- What a frozen layer is
- Why and how layers are frozen
- How to use Keras Applications for transfer learning
- How to adapt pretrained image models to new classification tasks

## Task 0 - Transfer Knowledge

File: `0-transfer.py`

The script trains a convolutional neural network to classify the CIFAR-10
dataset using transfer learning.

The implementation:

- Uses MobileNetV2 with ImageNet pretrained weights
- Resizes CIFAR-10 images from 32x32 to 160x160 with a Lambda layer
- Freezes the pretrained feature extractor during the first training stage
- Computes frozen features once and trains a new classification head on them
- Fine-tunes selected MobileNetV2 layers with a smaller learning rate
- Saves the compiled model as `cifar10.h5`

The file also contains:

```python
def preprocess_data(X, Y):
```

This function preprocesses CIFAR-10 images with MobileNetV2 preprocessing and
one-hot encodes the labels.

## Repository

- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/transfer_learning`
