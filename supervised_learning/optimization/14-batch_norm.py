#!/usr/bin/env python3
"""Creates a batch normalization layer in TensorFlow."""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a dense layer with batch normalization.

    Args:
        prev: Activated output of the previous layer.
        n: Number of nodes in the layer to create.
        activation: Activation function to apply.

    Returns:
        The activated output tensor of the layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(
        mode='fan_avg'
    )

    output = tf.keras.layers.Dense(
        units=n,
        activation=None,
        kernel_initializer=initializer
    )(prev)

    batch_norm = tf.keras.layers.BatchNormalization(
        epsilon=1e-7,
        gamma_initializer='ones',
        beta_initializer='zeros'
    )

    output = batch_norm(output, training=True)

    return activation(output)
