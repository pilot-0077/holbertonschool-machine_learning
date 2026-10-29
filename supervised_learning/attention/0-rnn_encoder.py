#!/usr/bin/env python3
"""RNN encoder for machine translation."""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """Encode a source sequence using an embedding layer and a GRU."""

    def __init__(self, vocab, embedding, units, batch):
        """Initialize the RNN encoder.

        Args:
            vocab: Size of the input vocabulary.
            embedding: Dimensionality of each embedding vector.
            units: Number of hidden units in the GRU.
            batch: Batch size.
        """
        super().__init__()
        self.batch = batch
        self.units = units
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

    def initialize_hidden_state(self):
        """Return zero-initialized hidden states of shape (batch, units)."""
        return tf.zeros((self.batch, self.units), dtype=tf.float32)

    def call(self, x, initial):
        """Encode token IDs and return the full outputs and final state.

        Args:
            x: Token IDs with shape (batch, input_seq_len).
            initial: Initial hidden state with shape (batch, units).

        Returns:
            A tuple containing:
              - outputs: (batch, input_seq_len, units)
              - hidden: (batch, units)
        """
        embedded = self.embedding(x)
        outputs, hidden = self.gru(
            embedded,
            initial_state=initial
        )
        return outputs, hidden
