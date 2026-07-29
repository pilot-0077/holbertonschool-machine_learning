#!/usr/bin/env python3
"""Creates an RMSprop optimizer in TensorFlow."""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """Creates a TensorFlow Keras RMSprop optimizer.

    Args:
        alpha: Learning rate.
        beta2: RMSprop decay rate.
        epsilon: Small constant for numerical stability.

    Returns:
        A TensorFlow Keras RMSprop optimizer.
    """
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )

    return optimizer
