#!/usr/bin/env python3
"""Creates a batch normalization layer."""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a dense layer followed by batch normalization.

    Args:
        prev: Activated output of the previous layer.
        n: Number of nodes in the new layer.
        activation: Activation function to apply.

    Returns:
        The activated output tensor of the new layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode='fan_avg'
    )

    dense = tf.keras.layers.Dense(
        units=n,
        activation=None,
        kernel_initializer=initializer
    )

    output = dense(prev)

    batch_norm = tf.keras.layers.BatchNormalization(
        axis=-1,
        epsilon=1e-8
    )

    normalized = batch_norm(output)

    return activation(normalized)
