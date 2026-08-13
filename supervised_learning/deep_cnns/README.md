# Deep Convolutional Neural Networks

This project explores modern deep convolutional neural network architectures
and techniques used to build and train very deep networks with TensorFlow and
Keras.

## Learning Objectives

- What a skip connection is
- What a bottleneck layer is
- What the Inception Network is
- What ResNet, ResNeXt, and DenseNet are
- How modern CNN architectures improve gradient flow and feature reuse
- How to reproduce a neural network architecture from a research paper

## Tasks

### 0. Inception Block

File: `0-inception_block.py`

Builds an Inception block with parallel 1x1, 3x3, 5x5, and max-pooling
branches followed by concatenation.

### 1. Inception Network

File: `1-inception_network.py`

Builds the Inception network for input images with shape `(224, 224, 3)` using
the Inception block from Task 0.

### 2. Identity Block

File: `2-identity_block.py`

Builds a ResNet identity block with a bottleneck structure and an identity
shortcut connection. Convolutional weights use He normal initialization with a
seed of 0.

### 3. Projection Block

File: `3-projection_block.py`

Builds a ResNet projection block. The shortcut path uses a 1x1 convolution so
that its dimensions match the main path before addition. Convolutional weights
use He normal initialization with a seed of 0.

### 4. ResNet-50

File: `4-resnet50.py`

Builds the ResNet-50 architecture using projection and identity blocks, ending
with average pooling and a 1000-class softmax output.

### 5. Dense Block

File: `5-dense_block.py`

Builds a DenseNet-B dense block using bottleneck layers. Each convolution is
preceded by batch normalization and ReLU activation. Weights use He normal
initialization with a seed of 0.

### 6. Transition Layer

File: `6-transition_layer.py`

Builds a DenseNet-C transition layer using batch normalization, ReLU, a 1x1
convolution for compression, and average pooling. Weights use He normal
initialization with a seed of 0.

### 7. DenseNet-121

File: `7-densenet121.py`

Builds the DenseNet-121 architecture with dense blocks of 6, 12, 24, and 16
layers separated by transition layers, followed by average pooling and a
1000-class softmax output.

## Repository

- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/deep_cnns`
