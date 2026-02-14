#!/usr/bin/env python3
"""Continuous posterior probability calculation"""
from scipy import special


def posterior(x, n, p1, p2):
    """
    Calculates the posterior probability that the probability of
    developing severe side effects falls within a specific range.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(p1, float) or not 0 <= p1 <= 1:
        raise ValueError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float) or not 0 <= p2 <= 1:
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    alpha = x + 1
    beta = n - x + 1

    # Compute the regularized incomplete beta function: I_p(x)
    return special.betainc(alpha, beta, p2) - special.betainc(alpha, beta, p1)
