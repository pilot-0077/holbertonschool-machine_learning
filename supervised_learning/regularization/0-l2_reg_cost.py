#!/usr/bin/env python3
"""Calculates the cost of a neural network with L2 regularization."""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost with L2 regularization.

    Args:
        cost: Cost of the network without regularization.
        lambtha: Regularization parameter.
        weights: Dictionary containing the network weights and biases.
        L: Number of layers in the neural network.
        m: Number of data points used.

    Returns:
        The cost of the network with L2 regularization.
    """
    l2_sum = 0

    for layer in range(1, L + 1):
        l2_sum += np.sum(np.square(weights["W{}".format(layer)]))

    return cost + (lambtha / (2 * m)) * l2_sum
