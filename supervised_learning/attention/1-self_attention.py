#!/usr/bin/env python3
"""Self-attention for machine translation."""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Calculate Bahdanau-style attention over encoder hidden states."""

    def __init__(self, units):
        """Initialize the alignment model.

        Args:
            units: Number of hidden units in the alignment model.
        """
        super().__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """Return the context vector and attention weights.

        Args:
            s_prev: Previous decoder state, shape (batch, units).
            hidden_states: Encoder outputs, shape
                (batch, input_seq_len, units).

        Returns:
            context: Weighted encoder states, shape (batch, units).
            weights: Attention weights, shape
                (batch, input_seq_len, 1).
        """
        decoder_state = tf.expand_dims(self.W(s_prev), axis=1)
        encoder_states = self.U(hidden_states)
        alignment = self.V(tf.tanh(decoder_state + encoder_states))
        weights = tf.nn.softmax(alignment, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights
