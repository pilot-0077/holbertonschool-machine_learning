#!/usr/bin/env python3
"""Calculates the likelihood of obtaining data given probabilities."""

import numpy as np


def likelihood(x, n, P):
    """Calculates the likelihood of obtaining x successes in n trials."""

    if not isinstance(n, int) or n <= 0:
        raise ValueError(
            "n must be a positive integer"
        )

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError(
            "P must be a 1D numpy.ndarray"
        )

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError(
            "All values in P must be in the range [0, 1]"
        )

    factorial_n = 1
    factorial_x = 1
    factorial_n_x = 1

    for i in range(1, n + 1):
        factorial_n *= i

    for i in range(1, x + 1):
        factorial_x *= i

    for i in range(1, n - x + 1):
        factorial_n_x *= i

    combination = factorial_n / (factorial_x * factorial_n_x)

    return combination * (P ** x) * ((1 - P) ** (n - x))
