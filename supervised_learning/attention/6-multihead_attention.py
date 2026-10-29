#!/usr/bin/env python3
"""Multi-head attention layer."""
import tensorflow as tf

sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Perform multi-head attention."""

    def __init__(self, dm, h):
        """Initialize the multi-head attention layer.

        Args:
            dm: Dimensionality of the model.
            h: Number of attention heads.
        """
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)
        self.linear = tf.keras.layers.Dense(units=dm)

    def call(self, Q, K, V, mask):
        """Calculate multi-head attention.

        Args:
            Q: Tensor of shape (batch, seq_len_q, dk).
            K: Tensor of shape (batch, seq_len_v, dk).
            V: Tensor of shape (batch, seq_len_v, dv).
            mask: Optional attention mask (the task specifies None).

        Returns:
            output: Tensor of shape (batch, seq_len_q, dm).
            weights: Tensor of shape (batch, h, seq_len_q, seq_len_v).
        """
        batch_size = tf.shape(Q)[0]
        seq_len_q = tf.shape(Q)[1]
        seq_len_v = tf.shape(K)[1]

        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        q = tf.reshape(q, (batch_size, seq_len_q, self.h, self.depth))
        k = tf.reshape(k, (batch_size, seq_len_v, self.h, self.depth))
        v = tf.reshape(v, (batch_size, seq_len_v, self.h, self.depth))

        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        scaled_attention, weights = sdp_attention(q, k, v, mask)

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, seq_len_q, self.dm)
        )
        output = self.linear(concat_attention)

        return output, weights
