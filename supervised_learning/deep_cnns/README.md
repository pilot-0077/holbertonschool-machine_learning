# Deep Convolutional Neural Networks

This project explores modern deep convolutional neural network architectures
and the techniques used to build and train very deep networks.

## Learning Objectives

By the end of this project, I should be able to explain:

- What a skip connection is
- What a bottleneck layer is
- What the Inception Network is
- What ResNet is
- What ResNeXt is
- What DenseNet is
- How modern CNN architectures improve gradient flow and feature reuse
- How to replicate a neural network architecture by reading a journal article
- How to translate an architecture diagram or table into TensorFlow/Keras code

## Key Concepts

### Skip Connections

Skip connections allow information to bypass one or more layers and be
combined with the output of deeper layers. They are a central idea behind
Residual Networks.

### Bottleneck Layers

Bottleneck layers commonly use `1x1` convolutions to reduce the number of
channels before more computationally expensive convolutions.

### Inception Networks

Inception networks process the same input through several parallel operations,
such as different convolution sizes and pooling, before concatenating their
outputs.

### ResNet

Residual Networks use skip connections and residual learning to make very
deep neural networks easier to train.

### ResNeXt

ResNeXt extends the residual-network idea by using multiple parallel
transformations inside a residual block.

### DenseNet

DenseNet connects each layer to all subsequent layers within a dense block,
encouraging feature reuse and improving gradient flow.

## Repository

- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/deep_cnns`
