#!/usr/bin/env python3
"""Builds a DenseNet dense block."""

from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Build a dense block using DenseNet-B bottleneck layers.

    Args:
        X: Output from the previous layer.
        nb_filters: Number of filters in X.
        growth_rate: Growth rate for the dense block.
        layers: Number of layers in the dense block.

    Returns:
        The block output and the number of filters in that output.
    """
    initializer = K.initializers.he_normal(seed=0)

    for _ in range(layers):
        batch1 = K.layers.BatchNormalization(axis=3)(X)
        relu1 = K.layers.Activation('relu')(batch1)

        bottleneck = K.layers.Conv2D(
            filters=4 * growth_rate,
            kernel_size=(1, 1),
            padding='same',
            kernel_initializer=initializer
        )(relu1)

        batch2 = K.layers.BatchNormalization(axis=3)(bottleneck)
        relu2 = K.layers.Activation('relu')(batch2)

        conv = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=initializer
        )(relu2)

        X = K.layers.Concatenate()([X, conv])
        nb_filters += growth_rate

    return X, nb_filters
