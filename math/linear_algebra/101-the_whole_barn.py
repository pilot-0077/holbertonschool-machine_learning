#!/usr/bin/env python3
"""Add two matrices of any dimension."""


def _shape(matrix):
    """Return the shape of a (nested-list) matrix."""
    if not isinstance(matrix, list):
        return []
    if matrix and isinstance(matrix[0], list):
        return [len(matrix)] + _shape(matrix[0])
    return [len(matrix)]


def _add(mat1, mat2):
    """Recursively add two matrices (same shape)."""
    if isinstance(mat1, list):
        return [_add(a, b) for a, b in zip(mat1, mat2)]
    return mat1 + mat2


def add_matrices(mat1, mat2):
    """Add two matrices; return None if shapes differ."""
    if _shape(mat1) != _shape(mat2):
        return None
    return _add(mat1, mat2)
