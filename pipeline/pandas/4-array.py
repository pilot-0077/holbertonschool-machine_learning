#!/usr/bin/env python3
"""Convert the last 10 rows of High and Close columns to a numpy array."""


def array(df):
    """
    Select the last 10 rows of High and Close columns and convert to numpy array.

    Args:
        df (pd.DataFrame): input dataframe with High and Close columns

    Returns:
        numpy.ndarray: selected values as numpy array
    """
    return df.loc[:, ["High", "Close"]].tail(10).to_numpy()
