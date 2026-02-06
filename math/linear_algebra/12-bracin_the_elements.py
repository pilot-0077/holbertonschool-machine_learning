#!/usr/bin/env python3
"""function for basic operations of matrices using numpy"""


def np_elementwise(mat1, mat2):
    """Performs element-wise addition, subtraction, multiplication, and division"""
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
