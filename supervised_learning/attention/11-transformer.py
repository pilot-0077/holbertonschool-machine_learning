#!/usr/bin/env python3
"""Transformer network."""
import tensorflow as tf

Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """Create a Transformer encoder-decoder network."""

    def __init__(
        self, N, dm, h, hidden, input_vocab, target_vocab,
        max_seq_input, max_seq_target, drop_rate=0.1
    ):
        """Initialize the Transformer model.

        Args:
            N: Number of encoder and decoder blocks.
            dm: Dimensionality of the model.
            h: Number of attention heads.
            hidden: Number of units in the feed-forward layers.
            input_vocab: Size of the input vocabulary.
            target_vocab: Size of the target vocabulary.
            max_seq_input: Maximum input sequence length.
            max_seq_target: Maximum target sequence length.
            drop_rate: Dropout rate.
        """
        super().__init__()
        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_input, drop_rate
        )
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_target, drop_rate
        )
        self.linear = tf.keras.layers.Dense(units=target_vocab)

    def call(
        self, inputs, target, training, encoder_mask,
        look_ahead_mask, decoder_mask
    ):
        """Run the encoder-decoder Transformer.

        Args:
            inputs: Source token IDs of shape (batch, input_seq_len).
            target: Target token IDs of shape (batch, target_seq_len).
            training: Whether the model is training.
            encoder_mask: Padding mask for encoder self-attention.
            look_ahead_mask: Look-ahead mask for decoder self-attention.
            decoder_mask: Padding mask for encoder-decoder attention.

        Returns:
            Logits of shape (batch, target_seq_len, target_vocab).
        """
        encoder_output = self.encoder(
            inputs, training=training, mask=encoder_mask
        )
        decoder_output = self.decoder(
            target,
            encoder_output,
            training=training,
            look_ahead_mask=look_ahead_mask,
            padding_mask=decoder_mask
        )
        return self.linear(decoder_output)
