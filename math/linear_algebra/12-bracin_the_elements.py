#!/usr/bin/env python3
"""function for basic operations of matrices using numpy"""


def np_elementwise(mat1, mat2):
    """Perform element-wise add, sub, mul, div and return them as a tuple."""
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return add, sub, mul, div
    
