#!/usr/bin/env python3
"""Calculates the cost of a Keras model with L2 regularization."""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """Calculates the cost with L2 regularization for each layer.

    Args:
        cost: Tensor containing the cost without L2 regularization.
        model: Keras model containing layers with L2 regularization.

    Returns:
        Tensor containing the cost for each regularized layer.
    """
    regularization_losses = tf.stack(model.losses)

    return cost + regularization_losses
