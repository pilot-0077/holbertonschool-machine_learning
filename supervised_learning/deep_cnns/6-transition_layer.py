#!/usr/bin/env python3
"""Builds a DenseNet transition layer."""

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """Build a DenseNet-C transition layer.

    Args:
        X: Output from the previous layer.
        nb_filters: Number of filters in X.
        compression: Compression factor for the transition layer.

    Returns:
        The transition-layer output and its number of filters.
    """
    initializer = K.initializers.he_normal(seed=0)
    nb_filters = int(nb_filters * compression)

    batch = K.layers.BatchNormalization(axis=3)(X)
    relu = K.layers.Activation('relu')(batch)

    conv = K.layers.Conv2D(
        filters=nb_filters,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(relu)

    output = K.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same'
    )(conv)

    return output, nb_filters
