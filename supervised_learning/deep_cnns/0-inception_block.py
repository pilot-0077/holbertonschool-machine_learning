#!/usr/bin/env python3
"""Builds an Inception block."""

from tensorflow import keras as K


def inception_block(A_prev, filters):
    """Build an Inception block.

    Args:
        A_prev: Output from the previous layer.
        filters: Tuple or list containing F1, F3R, F3, F5R, F5, and FPP.

    Returns:
        The concatenated output of the Inception block.
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    conv1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)

    conv3_reduce = K.layers.Conv2D(
        filters=F3R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)
    conv3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )(conv3_reduce)

    conv5_reduce = K.layers.Conv2D(
        filters=F5R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)
    conv5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=(5, 5),
        padding='same',
        activation='relu'
    )(conv5_reduce)

    pool = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same'
    )(A_prev)
    pool_proj = K.layers.Conv2D(
        filters=FPP,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(pool)

    return K.layers.Concatenate()([conv1, conv3, conv5, pool_proj])
