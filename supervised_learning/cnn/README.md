# Convolutional Neural Networks

This project covers forward and backward propagation through convolutional
and pooling layers, as well as building convolutional neural networks with
TensorFlow and Keras.

## Learning Objectives

- What a convolutional layer is
- What a pooling layer is
- Forward propagation through convolutional layers
- Forward propagation through pooling layers
- Back propagation through convolutional layers
- Back propagation through pooling layers
- How to build a CNN using TensorFlow and Keras

## Files

### `0-conv_forward.py`

Contains the function:

```python
def conv_forward(A_prev, W, b, activation, padding="same",
                 stride=(1, 1)):
