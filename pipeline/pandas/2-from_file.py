#!/usr/bin/env python3
"""Load data from a file into a pandas DataFrame."""

import pandas as pd


def from_file(filename, delimiter):
    """
    Load data from a file into a pandas DataFrame.

    Args:
        filename (str): path to the file
        delimiter (str): column separator

    Returns:
        pd.DataFrame: loaded data
    """
    return pd.read_csv(filename, sep=delimiter)
