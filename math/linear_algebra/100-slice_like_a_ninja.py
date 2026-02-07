#!/usr/bin/env python3
"""Slice a NumPy array along specific axes."""


def np_slice(matrix, axes=None):
    """Return a sliced copy of matrix according to axes."""
    if axes is None:
        axes = {}

    slicers = [slice(None)] * matrix.ndim
    for axis, spec in axes.items():
        slicers[axis] = slice(*spec)

    return matrix[tuple(slicers)].copy()
