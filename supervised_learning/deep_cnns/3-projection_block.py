#!/usr/bin/env python3
"""Builds a projection block for a deep residual network."""

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Build a projection block.

    Args:
        A_prev: Output from the previous layer.
        filters: Tuple or list containing F11, F3, and F12.
        s: Stride for the first convolution in the main path and shortcut.

    Returns:
        The activated output of the projection block.
    """
    F11, F3, F12 = filters

    conv1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(A_prev)
    batch1 = K.layers.BatchNormalization(axis=3)(conv1)
    relu1 = K.layers.Activation('relu')(batch1)

    conv2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(relu1)
    batch2 = K.layers.BatchNormalization(axis=3)(conv2)
    relu2 = K.layers.Activation('relu')(batch2)

    conv3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(relu2)
    batch3 = K.layers.BatchNormalization(axis=3)(conv3)

    shortcut = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    added = K.layers.Add()([batch3, shortcut])
    output = K.layers.Activation('relu')(added)

    return output
