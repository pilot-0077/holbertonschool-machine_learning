#!/usr/bin/env python3
"""Performs forward propagation over a convolutional layer."""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same",
                 stride=(1, 1)):
    """Perform forward propagation over a convolutional layer.

    Args:
        A_prev: NumPy array of shape (m, h_prev, w_prev, c_prev).
        W: NumPy array of shape (kh, kw, c_prev, c_new).
        b: NumPy array of shape (1, 1, 1, c_new).
        activation: Activation function applied to the convolution.
        padding: Either "same" or "valid".
        stride: Tuple (sh, sw) containing the convolution strides.

    Returns:
        The activated output of the convolutional layer.
    """
    m, h_prev, w_prev, _ = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil(
            ((h_prev - 1) * sh + kh - h_prev) / 2
        ))
        pw = int(np.ceil(
            ((w_prev - 1) * sw + kw - w_prev) / 2
        ))
    else:
        ph = 0
        pw = 0

    A_prev_padded = np.pad(
        A_prev,
        (
            (0, 0),
            (ph, ph),
            (pw, pw),
            (0, 0)
        ),
        mode="constant"
    )

    h_new = ((h_prev + (2 * ph) - kh) // sh) + 1
    w_new = ((w_prev + (2 * pw) - kw) // sw) + 1

    Z = np.zeros((m, h_new, w_new, c_new))
    biases = b.reshape((1, c_new))

    for i in range(h_new):
        for j in range(w_new):
            row_start = i * sh
            col_start = j * sw

            region = A_prev_padded[
                :,
                row_start:row_start + kh,
                col_start:col_start + kw,
                :
            ]

            Z[:, i, j, :] = np.sum(
                region[:, :, :, :, np.newaxis] * W,
                axis=(1, 2, 3)
            ) + biases

    return activation(Z)
