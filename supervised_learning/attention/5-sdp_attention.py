#!/usr/bin/env python3
"""Scaled dot product attention."""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Calculate scaled dot product attention.

    Args:
        Q: Query tensor with shape (..., seq_len_q, dk).
        K: Key tensor with shape (..., seq_len_v, dk).
        V: Value tensor with shape (..., seq_len_v, dv).
        mask: Optional mask broadcastable to (..., seq_len_q, seq_len_v).

    Returns:
        tuple: The attention output and attention weights.
    """
    dk = tf.cast(tf.shape(K)[-1], Q.dtype)
    scores = tf.matmul(Q, K, transpose_b=True)
    scores = scores / tf.math.sqrt(dk)

    if mask is not None:
        mask = tf.cast(mask, scores.dtype)
        scores += mask * tf.cast(-1e9, scores.dtype)

    weights = tf.nn.softmax(scores, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights
