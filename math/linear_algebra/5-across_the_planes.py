#!/usr/bin/env python3
"""Add two 2D matrices element-wise."""


def add_matrices2D(mat1, mat2):
    """Return a new 2D matrix that is the element-wise sum of mat1 and mat2."""
    if len(mat1) != len(mat2):
        return None

    new = []
    for row1, row2 in zip(mat1, mat2):
        if len(row1) != len(row2):
            return None
        new.append([a + b for a, b in zip(row1, row2)])

    return new
