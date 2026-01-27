#!/usr/bin/env python3
"""Slice a DataFrame every 60 rows for selected columns."""


def slice(df):
    """
    Extract High, Low, Close and Volume_(BTC) columns
    and select every 60th row.

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: sliced dataframe
    """
    columns = ["High", "Low", "Close", "Volume_(BTC)"]
    return df.loc[:, columns].iloc[::60]
