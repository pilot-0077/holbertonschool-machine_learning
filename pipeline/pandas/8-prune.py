#!/usr/bin/env python3
"""Prune rows with NaN Close values"""


def prune(df):
    """Removes rows where Close is NaN"""
    return df.dropna(subset=["Close"])
