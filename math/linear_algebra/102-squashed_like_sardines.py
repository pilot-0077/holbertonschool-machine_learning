#!/usr/bin/env python3
"""Concatenate two matrices along a specific axis."""


def _shape(matrix):
    """Return shape of nested-list matrix."""
    if not isinstance(matrix, list):
        return []
    if matrix and isinstance(matrix[0], list):
        return [len(matrix)] + _shape(matrix[0])
    return [len(matrix)]


def _cat(mat1, mat2, axis, depth=0):
    """Recursively concatenate along axis."""
    if depth == axis:
        # concatenate at this level (make a new list)
        return [x for x in mat1] + [x for x in mat2]

    # otherwise, go deeper row by row
    return [_cat(a, b, axis, depth + 1) for a, b in zip(mat1, mat2)]


def cat_matrices(mat1, mat2, axis=0):
    """Concatenate two matrices along axis; return None if impossible."""
    s1 = _shape(mat1)
    s2 = _shape(mat2)

    # must have same number of dimensions
    if len(s1) != len(s2):
        return None

    # axis must be valid
    if axis < 0 or axis >= len(s1):
        return None

    # all dims except the concatenation axis must match
    for i in range(len(s1)):
        if i != axis and s1[i] != s2[i]:
            return None

    return _cat(mat1, mat2, axis)
