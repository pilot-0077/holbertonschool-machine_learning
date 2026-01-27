#!/usr/bin/env python3
"""Sort a DataFrame in reverse order and transpose it."""


def flip_switch(df):
    """
    Sort the DataFrame in reverse chronological order
    and transpose it.

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: transformed dataframe
    """
    df = df.sort_values(by="Timestamp", ascending=False)
    return df.transpose()
