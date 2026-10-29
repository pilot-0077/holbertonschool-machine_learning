#!/usr/bin/env python3
"""RNN decoder for machine translation."""
import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Decode target tokens using a GRU and attention."""

    def __init__(self, vocab, embedding, units, batch):
        """Initialize the decoder layers.

        Args:
            vocab: Size of the target vocabulary.
            embedding: Dimensionality of the embedding vectors.
            units: Number of GRU hidden units.
            batch: Batch size.
        """
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
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """Return target-vocabulary scores and the updated hidden state.

        Args:
            x: Previous target token IDs, shape (batch, 1).
            s_prev: Previous decoder state, shape (batch, units).
            hidden_states: Encoder outputs, shape
                (batch, input_seq_len, units).

        Returns:
            y: Output scores, shape (batch, vocab).
            s: Updated decoder hidden state, shape (batch, units).
        """
        context, _ = self.attention(s_prev, hidden_states)
        x = tf.cast(x, context.dtype)
        decoder_input = tf.concat([context, x], axis=1)
        decoder_input = tf.expand_dims(decoder_input, axis=1)
        outputs, state = self.gru(
            decoder_input,
            initial_state=s_prev
        )
        outputs = tf.squeeze(outputs, axis=1)
        y = self.F(outputs)
        return y, state
