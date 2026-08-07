#!/usr/bin/env python3
"""Builds a modified LeNet-5 architecture using Keras."""

from tensorflow import keras as K


def lenet5(X):
    """Build a modified LeNet-5 neural network.

    Args:
        X: Keras Input of shape (m, 28, 28, 1) containing input images.

    Returns:
        A compiled Keras Model using Adam optimization and accuracy metrics.
    """
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(X)

    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)

    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(pool1)

    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv2)

    flatten = K.layers.Flatten()(pool2)

    dense1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(flatten)

    dense2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(dense1)

    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(dense2)

    model = K.Model(inputs=X, outputs=output)

    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
