#!/usr/bin/env python3
"""Creates a neural network layer with L2 regularization."""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a dense layer with L2 regularization.

    Args:
        prev: Tensor containing the output of the previous layer.
        n: Number of nodes in the new layer.
        activation: Activation function for the new layer.
        lambtha: L2 regularization parameter.

    Returns:
        The output tensor of the new layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode="fan_avg"
    )

    regularizer = tf.keras.regularizers.L2(lambtha)

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer
    )

    return layer(prev)
