#!/usr/bin/env python3
"""RNN decoder for machine translation."""
import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Decode target tokens using a GRU and attention."""

    def __init__(self, vocab, embedding, units, batch):
        """Initialize the decoder layers."""
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )
        self.gru = tf.keras.layers.GRU(
            units=units,
            recurrent_initializer='glorot_uniform',
            return_sequences=True,
            return_state=True
        )
        self.F = tf.keras.layers.Dense(units=vocab)

    def call(self, x, s_prev, hidden_states):
        """Return output scores and the updated decoder state."""
        embedded = self.embedding(x)
        attention = SelfAttention(s_prev.shape[-1])
        context, _ = attention(s_prev, hidden_states)
        context = tf.expand_dims(context, axis=1)
        decoder_input = tf.concat([context, embedded], axis=-1)
        outputs, state = self.gru(
            decoder_input,
            initial_state=s_prev
        )
        outputs = self.F(outputs)
        y = tf.squeeze(outputs, axis=1)
        return y, state
