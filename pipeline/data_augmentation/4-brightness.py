#!/usr/bin/env python3
"""Randomly changes the brightness of an image."""

import tensorflow as tf


def change_brightness(image, max_delta):
    """Randomly change the brightness of an image.

    Args:
        image: A 3D tf.Tensor containing the image.
        max_delta: Maximum amount to brighten or darken the image.

    Returns:
        The image with randomly adjusted brightness.
    """
    return tf.image.random_brightness(image, max_delta)
