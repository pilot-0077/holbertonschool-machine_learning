#!/usr/bin/env python3
"""Bayesian Information Criterion for Gaussian mixture models."""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Find the best number of GMM clusters using BIC.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        kmin: Minimum number of clusters to test, inclusive.
        kmax: Maximum number of clusters to test, inclusive.
        iterations: Maximum number of EM iterations.
        tol: Non-negative tolerance used by EM.
        verbose: Whether EM should print progress information.

    Returns:
        best_k, best_result, l, b, or four None values on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None

    n, d = X.shape
    if n == 0 or d == 0:
        return None, None, None, None

    if not isinstance(kmin, int) or isinstance(kmin, bool) or kmin <= 0:
        return None, None, None, None

    if kmax is None:
        kmax = n
    elif (not isinstance(kmax, int) or isinstance(kmax, bool)
          or kmax <= 0):
        return None, None, None, None

    if kmax <= kmin or kmin > n or kmax > n:
        return None, None, None, None

    if (not isinstance(iterations, int) or isinstance(iterations, bool)
            or iterations <= 0):
        return None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    count = kmax - kmin + 1
    likelihoods = np.empty(count)
    bics = np.empty(count)
    results = []

    for index, k in enumerate(range(kmin, kmax + 1)):
        try:
            pi, m, S, g, li = expectation_maximization(
                X, k, iterations, tol, verbose
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return None, None, None, None

        if (pi is None or m is None or S is None or
                g is None or li is None):
            return None, None, None, None

        parameters = (
            k * d
            + k * d * (d + 1) // 2
            + k - 1
        )
        bic = parameters * np.log(n) - 2 * li

        results.append((pi, m, S))
        likelihoods[index] = li
        bics[index] = bic

    best_index = np.argmin(bics)
    best_k = kmin + best_index
    best_result = results[best_index]

    return best_k, best_result, likelihoods, bics
