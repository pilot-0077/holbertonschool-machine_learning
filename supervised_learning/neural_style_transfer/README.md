# Neural Style Transfer

This project covers Neural Style Transfer (NST), a technique that combines the
content of one image with the visual style of another using pretrained
convolutional neural networks.

## Learning Objectives

By the end of this project, I should be able to explain:

- What Neural Style Transfer is
- What a Gram matrix is and why it represents visual style
- How to calculate content cost
- How to calculate style cost
- How content and style losses are combined
- What TensorFlow GradientTape is and how it is used
- How to optimize an image while keeping the pretrained network frozen
- How to perform a complete Neural Style Transfer pipeline

## Core Concepts

Neural Style Transfer typically uses a pretrained CNN such as VGG19 to extract
features from content, style, and generated images. Content similarity is
measured from feature activations, while style similarity is measured using
Gram matrices. The generated image is iteratively updated with gradients to
minimize the combined content and style loss.

## Repository

- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/neural_style_transfer`
