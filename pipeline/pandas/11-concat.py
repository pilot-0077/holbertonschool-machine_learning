#!/usr/bin/env python3
"""Concatenate bitstamp and coinbase dataframes with keys."""

import pandas as pd

index = __import__('10-index').index


def concat(df1, df2):
    """
    Index both dataframes on Timestamp, take df2 rows up to 1417411920,
    then concatenate df2 on top of df1 using keys.
    """
    df1 = index(df1)
    df2 = index(df2)

    df2 = df2.loc[:1417411920]

    return pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
