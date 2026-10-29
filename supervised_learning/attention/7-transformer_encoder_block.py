#!/usr/bin/env python3
"""Transformer encoder block."""
import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """Create a Transformer encoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize the encoder block.

        Args:
            dm: Dimensionality of the model.
            h: Number of attention heads.
            hidden: Number of units in the feed-forward hidden layer.
            drop_rate: Dropout rate.
        """
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(
            units=hidden, activation='relu'
        )
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Run the input through the encoder block.

        Args:
            x: Input tensor of shape (batch, input_seq_len, dm).
            training: Whether the model is training.
            mask: Optional mask for multi-head attention.

        Returns:
            Tensor of shape (batch, input_seq_len, dm).
        """
        attention_output, _ = self.mha(x, x, x, mask)
        attention_output = self.dropout1(
            attention_output, training=training
        )
        out1 = self.layernorm1(x + attention_output)

        feed_forward = self.dense_hidden(out1)
        feed_forward = self.dense_output(feed_forward)
        feed_forward = self.dropout2(feed_forward, training=training)
        output = self.layernorm2(out1 + feed_forward)

        return output
