#!/usr/bin/env python3
"""Fill missing values and drop Weighted_Price column."""


def fill(df):
    """
    Remove Weighted_Price column and fill missing values:
    - Close: forward fill
    - High, Low, Open: fill with same row's Close
    - Volume_(BTC), Volume_(Currency): fill with 0

    Returns:
        pd.DataFrame: modified dataframe
    """
    df = df.drop(columns=["Weighted_Price"], errors="ignore")

    df["Close"] = df["Close"].fillna(method="ffill")

    for col in ("High", "Low", "Open"):
        df[col] = df[col].fillna(df["Close"])

    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

    return df
