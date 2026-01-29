#!/usr/bin/env python3

import pandas as pd


def fill(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in the DataFrame.

- Remove the 'Weighted_Price' column if present.
- Forward-fill missing values in 'Close'.
- Fill missing values in 'High', 'Low', 'Open' with the same row's 'Close'.
- Set missing values in 'Volume_(BTC)' and 'Volume_(Currency)' to 0.

Returns:
    The modified DataFrame.
"""
    
df.drop(columns=['Weighted_Price'], inplace=True, errors='ignore')
df['Close'].fillna(method='ffill', inplace=True)

for col in ('Open', 'High', 'Low'):
    df[col].fillna(value=df['Close'], inplace=True)
df['Volume_(BTC)'].fillna(0, inplace=True)
df['Volume_(Currency)'].fillna(0, inplace=True)

return df
