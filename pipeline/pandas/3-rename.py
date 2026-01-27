#!/usr/bin/env python3
"""Rename Timestamp to Datetime and select Datetime and Close columns."""

import pandas as pd


def rename(df):
    """
    Rename the Timestamp column to Datetime, convert it to datetime,
    and keep only Datetime and Close columns.

    Args:
        df (pd.DataFrame): input dataframe with a Timestamp column

    Returns:
        pd.DataFrame: modified dataframe
    """
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    return df.loc[:, ["Datetime", "Close"]]
