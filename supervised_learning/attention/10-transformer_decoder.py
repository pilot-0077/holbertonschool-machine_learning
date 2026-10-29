#!/usr/bin/env python3
"""Transformer decoder."""
import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """Create a Transformer decoder."""

    def __init__(
        self, N, dm, h, hidden, target_vocab, max_seq_len, drop_rate=0.1
    ):
        """Initialize the Transformer decoder.

        Args:
            N: Number of decoder blocks.
            dm: Dimensionality of the model.
            h: Number of attention heads.
            hidden: Number of units in the feed-forward hidden layer.
            target_vocab: Size of the target vocabulary.
            max_seq_len: Maximum supported target sequence length.
            drop_rate: Dropout rate.
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate)
            for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(
        self, x, encoder_output, training, look_ahead_mask, padding_mask
    ):
        """Decode target token IDs using the encoder output.

        Args:
            x: Target token IDs of shape (batch, target_seq_len).
            encoder_output: Encoder output of shape (batch, input_seq_len, dm).
            training: Whether the model is training.
            look_ahead_mask: Mask for decoder self-attention.
            padding_mask: Mask for encoder-decoder attention.

        Returns:
            Tensor of shape (batch, target_seq_len, dm).
        """
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, x.dtype))
        x += tf.cast(self.positional_encoding[:seq_len], x.dtype)
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(
                x,
                encoder_output,
                training,
                look_ahead_mask,
                padding_mask
            )

        return x
