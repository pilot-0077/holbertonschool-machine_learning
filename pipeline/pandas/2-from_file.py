#!/usr/bin/env python3
"""Load data from a file into a pandas DataFrame."""

import pandas as pd

df = pd.read_csv("filename.csv")

def from_file(filename, delimiter):
    """
    function from_file
    Args:
        filename: file to load from
        delimiter: column separator
    Returns: loaded pd.DataFrame
    """
    return pd.read_csv(filename, sep=delimiter)
