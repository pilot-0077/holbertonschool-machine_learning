#!/usr/bin/env python3
"""Creates an Adam optimizer in TensorFlow."""

import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """Creates a TensorFlow Keras Adam optimizer.

    Args:
        alpha: Learning rate.
        beta1: Exponential decay rate for the first moment.
        beta2: Exponential decay rate for the second moment.
        epsilon: Small constant for numerical stability.

    Returns:
        A TensorFlow Keras Adam optimizer.
    """
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )

    return optimizer
