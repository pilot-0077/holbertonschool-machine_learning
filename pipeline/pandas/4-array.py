#!/usr/bin/env python3
"""Convert the last 10 rows of High and Close columns
to a numpy array.
"""


def array(df):
    """
    Select the last 10 rows of High and Close columns
    and convert them to a numpy array.

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        numpy.ndarray: selected values
    """
    return df.loc[:, ["High", "Close"]].tail(10).to_numpy()
