#!/usr/bin/env python3
"""Create padding and look-ahead masks for Transformer training."""
import tensorflow as tf


def create_masks(inputs, target):
    """Create the encoder, combined decoder, and decoder padding masks.

    Args:
        inputs: Tensor of shape (batch_size, seq_len_in) containing source
            token IDs. Padding tokens are represented by 0.
        target: Tensor of shape (batch_size, seq_len_out) containing target
            token IDs. Padding tokens are represented by 0.

    Returns:
        A tuple containing:
            encoder_mask: Source padding mask with shape
                (batch_size, 1, 1, seq_len_in).
            combined_mask: Look-ahead and target padding mask with shape
                (batch_size, 1, seq_len_out, seq_len_out).
            decoder_mask: Source padding mask with shape
                (batch_size, 1, 1, seq_len_in).
    """
    # Mask padding tokens in the input sequence.
    input_padding = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = input_padding[:, tf.newaxis, tf.newaxis, :]

    # Mask padding tokens in the target sequence.
    target_padding = tf.cast(tf.math.equal(target, 0), tf.float32)
    target_padding = target_padding[:, tf.newaxis, tf.newaxis, :]

    # Mask future target tokens (upper triangular part above the diagonal).
    target_length = tf.shape(target)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((target_length, target_length), dtype=tf.float32),
        -1,
        0
    )

    # Combine target padding and look-ahead masks.
    combined_mask = tf.maximum(target_padding, look_ahead_mask)

    # The decoder's second attention block masks source padding positions.
    decoder_mask = input_padding[:, tf.newaxis, tf.newaxis, :]

    return encoder_mask, combined_mask, decoder_mask
