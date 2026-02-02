#!/usr/bin/env python3
"""Set Timestamp column as the index of a DataFrame."""


def index(df):
    """
    Set the Timestamp column as the index.

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: modified dataframe
    """
    return df.set_index("Timestamp")
