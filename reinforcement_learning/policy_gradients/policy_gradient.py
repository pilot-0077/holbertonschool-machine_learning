#!/usr/bin/env python3
"""Simple policy function for policy gradient methods."""

import numpy as np


def policy(matrix, weight):
    """Compute the policy probabilities for a state matrix.

    Args:
        matrix: numpy.ndarray containing the state observations.
        weight: numpy.ndarray containing the policy weights.

    Returns:
        A numpy.ndarray containing the action probabilities.
    """
    logits = np.matmul(matrix, weight)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)
