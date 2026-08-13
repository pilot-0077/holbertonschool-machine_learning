#!/usr/bin/env python3
"""Builds the ResNet-50 architecture using Keras."""

from tensorflow import keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Build the ResNet-50 architecture.

    Returns:
        A Keras model implementing ResNet-50.
    """
    X = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(X)
    batch1 = K.layers.BatchNormalization(axis=3)(conv1)
    relu1 = K.layers.Activation('relu')(batch1)

    pool1 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(relu1)

    # Stage 2
    A = projection_block(pool1, [64, 64, 256], s=1)
    A = identity_block(A, [64, 64, 256])
    A = identity_block(A, [64, 64, 256])

    # Stage 3
    A = projection_block(A, [128, 128, 512], s=2)
    A = identity_block(A, [128, 128, 512])
    A = identity_block(A, [128, 128, 512])
    A = identity_block(A, [128, 128, 512])

    # Stage 4
    A = projection_block(A, [256, 256, 1024], s=2)
    A = identity_block(A, [256, 256, 1024])
    A = identity_block(A, [256, 256, 1024])
    A = identity_block(A, [256, 256, 1024])
    A = identity_block(A, [256, 256, 1024])
    A = identity_block(A, [256, 256, 1024])

    # Stage 5
    A = projection_block(A, [512, 512, 2048], s=2)
    A = identity_block(A, [512, 512, 2048])
    A = identity_block(A, [512, 512, 2048])

    avg_pool = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1),
        padding='valid'
    )(A)

    output = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(avg_pool)

    return K.models.Model(inputs=X, outputs=output)
