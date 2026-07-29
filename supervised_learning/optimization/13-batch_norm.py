#!/usr/bin/env python3
"""Normalizes an unactivated layer using batch normalization."""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalizes an unactivated output using batch normalization.

    Args:
        Z: numpy.ndarray of shape (m, n) containing unactivated outputs.
        gamma: numpy.ndarray of shape (1, n) containing scale parameters.
        beta: numpy.ndarray of shape (1, n) containing offset parameters.
        epsilon: Small value used to avoid division by zero.

    Returns:
        The batch-normalized output.
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)

    Z_normalized = (Z - mean) / np.sqrt(variance + epsilon)

    return gamma * Z_normalized + beta
