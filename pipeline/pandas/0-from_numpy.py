#!/usr/bin/env python3
"""
Creates a pandas DataFrame from a NumPy ndarray.
"""

import pandas as pd


def from_numpy(array):
    """
    Creates a pandas DataFrame from a NumPy ndarray.

    Columns are labeled alphabetically in uppercase letters.

    Args:
        array (np.ndarray): NumPy array to convert

    Returns:
        pd.DataFrame: DataFrame with labeled columns
    """
    cols = [chr(ord('A') + i) for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=cols)
