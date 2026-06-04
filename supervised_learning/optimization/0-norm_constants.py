#!/usr/bin/env python3
"""Calculates normalization constants for a matrix."""

import numpy as np


def normalization_constants(X):
    """
    Calculates the mean and standard deviation of each feature.

    Args:
        X: numpy.ndarray of shape (m, nx)

    Returns:
        mean, standard deviation
    """

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return mean, std
