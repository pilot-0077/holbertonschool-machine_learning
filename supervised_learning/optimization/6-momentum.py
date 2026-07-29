#!/usr/bin/env python3
"""Creates a momentum optimization operation in TensorFlow."""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """Creates a momentum optimizer training operation.

    Args:
        loss: Loss tensor of the network.
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
