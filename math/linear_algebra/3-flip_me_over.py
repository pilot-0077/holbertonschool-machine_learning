#!/usr/bin/env python3
"""Transpose a 2D matrix."""


def matrix_transpose(matrix):
    """Return the transpose of a 2D matrix."""
    rows = len(matrix)
    cols = len(matrix[0])
    return [[matrix[r][c] for r in range(rows)] for c in range(cols)]
