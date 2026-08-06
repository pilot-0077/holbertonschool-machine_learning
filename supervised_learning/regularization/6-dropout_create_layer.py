#!/usr/bin/env python3
"""Creates a neural network layer using dropout."""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """Creates a dense layer followed by dropout.

    Args:
        prev: Tensor containing the output of the previous layer.
        n: Number of nodes in the new layer.
        activation: Activation function for the new layer.
        keep_prob: Probability that a node is kept.
        training: Whether the layer is operating in training mode.

    Returns:
        The output tensor of the layer after dropout.
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode="fan_avg"
    )

    dense = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )

    output = dense(prev)

    dropout = tf.keras.layers.Dropout(
        rate=1 - keep_prob
    )

    return dropout(output, training=training)
