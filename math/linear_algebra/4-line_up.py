#!/usr/bin/env python3
"""Add two arrays element-wise."""


def add_arrays(arr1, arr2):
    """Return a new list that is the element-wise sum of arr1 and arr2."""
    if len(arr1) != len(arr2):
        return None
    return [a + b for a, b in zip(arr1, arr2)]
