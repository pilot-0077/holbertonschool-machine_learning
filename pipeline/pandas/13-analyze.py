#!/usr/bin/env python3
"""Compute descriptive statistics for all columns except Timestamp."""


def analyze(df):
    """
    Compute descriptive statistics for all columns except Timestamp.

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: statistics dataframe
    """
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])
    return df.describe()
