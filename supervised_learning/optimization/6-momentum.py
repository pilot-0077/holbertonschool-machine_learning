#!/usr/bin/env python3
"""Creates a Keras optimizer using gradient descent with momentum."""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Creates an SGD optimizer with momentum.

    Args:
        alpha: Learning rate.
        beta1: Momentum weight.

    Returns:
        A TensorFlow Keras SGD optimizer.
    """
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=alpha,
        momentum=beta1
    )

    return optimizer
