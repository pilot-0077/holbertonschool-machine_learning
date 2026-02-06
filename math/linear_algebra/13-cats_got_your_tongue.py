#!/usr/bin/env python3
"""Concatenate two matrices along a specific axis using NumPy."""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Return the concatenation of mat1 and mat2 along axis."""
    return np.concatenate((mat1, mat2), axis=axis)
