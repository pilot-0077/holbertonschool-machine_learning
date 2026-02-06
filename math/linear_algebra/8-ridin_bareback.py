#!/usr/bin/env python3
"""Matrix multiplication."""


def mat_mul(mat1, mat2):
    """Perform matrix multiplication (mat1 x mat2)."""
    if len(mat1[0]) != len(mat2):
        return None

    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            s = 0
            for k in range(len(mat2)):
                s += mat1[i][k] * mat2[k][j]
            row.append(s)
        result.append(row)

    return result
