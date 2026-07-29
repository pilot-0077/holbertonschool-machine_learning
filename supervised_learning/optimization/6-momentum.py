#!/usr/bin/env python3
"""Creates a TensorFlow momentum optimization operation."""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """Creates the training operation using momentum optimization.

    Args:
        loss: Loss of the neural network.
        alpha: Learning rate.
        beta1: Momentum weight.

    Returns:
        The momentum optimization operation.
    """
    optimizer = tf.compat.v1.train.MomentumOptimizer(
        learning_rate=alpha,
        momentum=beta1
    )

    return optimizer.minimize(loss)
