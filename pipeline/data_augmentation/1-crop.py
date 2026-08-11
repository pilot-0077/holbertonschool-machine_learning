#!/usr/bin/env python3
"""Performs a random crop of an image."""

import tensorflow as tf


def crop_image(image, size):
    """Perform a random crop of an image.

    Args:
        image: A 3D tf.Tensor containing the image to crop.
        size: A tuple containing the size of the crop.

    Returns:
        The randomly cropped image.
    """
    return tf.image.random_crop(image, size=size)
