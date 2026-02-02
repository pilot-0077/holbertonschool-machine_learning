#!/usr/bin/env python3
"""Concatenate two tables with Timestamp as first MultiIndex level."""

import pandas as pd

index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Concatenate bitstamp and coinbase between timestamps 1417411980 and
    1417417980 inclusive, then rearrange MultiIndex so Timestamp is first.
    """
    df1 = index(df1)
    df2 = index(df2)

    start = 1417411980
    end = 1417417980

    df1 = df1.loc[start:end]
    df2 = df2.loc[start:end]

    df = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])

    df = df.swaplevel(0, 1, axis=0)
    df = df.sort_index()

    return df
