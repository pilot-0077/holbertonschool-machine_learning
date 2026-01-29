#!/usr/bin/env python3

import pandas as pd


def fill(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in the DataFrame.

    - Remove the 'Weighted_Price' column if present.
    - Forward-fill missing values in 'Close'.
    - Fill missing values in 'High', 'Low', 'Open' with the same row's 'Close'.
    - Set missing values in 'Volume_(BTC)' and 'Volume_(Currency)' to 0.

    Returns:
        The modified pd.DataFrame.
    """

    # Remove the Weighted_Price column if it exists
    df.drop(columns=['Weighted_Price'], inplace=True, errors='ignore')

    # Fill Close with previous row's value
    df['Close'].fillna(method='ffill', inplace=True)

    # Fill Open, High, Low with the same row's Close value
    for col in ('Open', 'High', 'Low'):
        df[col].fillna(value=df['Close'], inplace=True)

    # Set missing volumes to 0
    df['Volume_(BTC)'].fillna(0, inplace=True)
    df['Volume_(Currency)'].fillna(0, inplace=True)

    return df
