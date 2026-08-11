#!/usr/bin/env python3
"""Contains a function for horizontally flipping an image."""

import tensorflow as tf


def flip_image(image):
    """Flip an image horizontally.

    Args:
        image: A 3D tf.Tensor containing the image.

    Returns:
        The horizontally flipped image.
    """
    return tf.image.flip_left_right(image)
