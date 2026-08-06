#!/usr/bin/env python3
"""Performs a convolution on images with channels."""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on images with multiple channels.

    Args:
        images: NumPy array with shape (m, h, w, c).
        kernel: NumPy array with shape (kh, kw, c).
        padding: Either 'same', 'valid', or a tuple (ph, pw).
        stride: Tuple (sh, sw) containing the strides.

    Returns:
        NumPy array containing the convolved images.
    """
    m, h, w, _ = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride

    if padding == 'same':
        pad_h = int(np.ceil(
            ((h - 1) * sh + kh - h) / 2
        ))
        pad_w = int(np.ceil(
            ((w - 1) * sw + kw - w) / 2
        ))
    elif padding == 'valid':
        pad_h = 0
        pad_w = 0
    elif isinstance(padding, tuple):
        pad_h, pad_w = padding

    images_padded = np.pad(
        images,
        (
            (0, 0),
            (pad_h, pad_h),
            (pad_w, pad_w),
            (0, 0)
        ),
        mode='constant'
    )

    output_h = ((h + (2 * pad_h) - kh) // sh) + 1
    output_w = ((w + (2 * pad_w) - kw) // sw) + 1

    convolved = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            row_start = i * sh
            col_start = j * sw

            region = images_padded[
                :,
                row_start:row_start + kh,
                col_start:col_start + kw,
                :
            ]

            convolved[:, i, j] = np.sum(
                region * kernel,
                axis=(1, 2, 3)
            )

    return convolved
