#!/usr/bin/env python3
"""Sort DataFrame by High price in descending order."""


def high(df):
    """
    Sort the DataFrame by the High column in descending order.

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: sorted dataframe
    """
    return df.sort_values(by="High", ascending=False)
