#!/usr/bin/env python3
"""Positional encoding for a Transformer."""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """Calculate sinusoidal positional encodings.

    Args:
        max_seq_len (int): Maximum sequence length.
        dm (int): Model depth.

    Returns:
        numpy.ndarray: Positional encoding matrix of shape
            (max_seq_len, dm).
    """
    positions = np.arange(max_seq_len)[:, np.newaxis]
    dimensions = np.arange(dm)[np.newaxis, :]

    angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / dm)
    angle_rads = positions * angle_rates

    encoding = np.zeros((max_seq_len, dm))
    encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return encoding
