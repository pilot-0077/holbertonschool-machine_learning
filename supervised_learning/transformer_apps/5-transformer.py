#!/usr/bin/env python3
"""Transformer encoder-decoder model for machine translation."""
import tensorflow as tf


def positional_encoding(max_seq_len, dm):
    """Create sinusoidal positional encodings using TensorFlow operations.

    Args:
        max_seq_len: Maximum supported sequence length.
        dm: Dimensionality of the model.

    Returns:
        A float tensor with shape (max_seq_len, dm).
    """
    positions = tf.cast(tf.range(max_seq_len)[:, tf.newaxis], tf.float32)
    dimensions = tf.cast(tf.range(dm)[tf.newaxis, :], tf.float32)
    exponent = (2 * tf.floor(dimensions / 2)) / tf.cast(dm, tf.float32)
    angle_rates = tf.pow(10000.0, -exponent)
    angles = positions * angle_rates

    even_dimensions = tf.equal(
        tf.math.floormod(tf.range(dm), 2), 0
    )
    return tf.where(even_dimensions[tf.newaxis, :],
                    tf.sin(angles), tf.cos(angles))


def sdp_attention(Q, K, V, mask=None):
    """Calculate scaled dot-product attention.

    Args:
        Q: Query tensor.
        K: Key tensor.
        V: Value tensor.
        mask: Optional padding or look-ahead mask.

    Returns:
        The attention output and the attention weights.
    """
    scores = tf.matmul(Q, K, transpose_b=True)
    depth = tf.cast(tf.shape(K)[-1], scores.dtype)
    scores = scores / tf.math.sqrt(depth)

    if mask is not None:
        mask = tf.cast(mask, scores.dtype)
        scores += mask * tf.cast(-1e9, scores.dtype)

    weights = tf.nn.softmax(scores, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """Perform multi-head attention."""

    def __init__(self, dm, h):
        """Initialize the projection layers and attention heads."""
        super().__init__()
        self.dm = dm
        self.h = h
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Split the final model dimension into attention heads."""
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask=None):
        """Apply multi-head attention and return output and weights."""
        batch_size = tf.shape(Q)[0]
        q = self.split_heads(self.Wq(Q), batch_size)
        k = self.split_heads(self.Wk(K), batch_size)
        v = self.split_heads(self.Wv(V), batch_size)

        attention, weights = sdp_attention(q, k, v, mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        attention = tf.reshape(attention, (batch_size, -1, self.dm))
        return self.linear(attention), weights


class EncoderBlock(tf.keras.layers.Layer):
    """A Transformer encoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize self-attention, feed-forward, and normalization layers."""
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Run self-attention and the position-wise feed-forward network."""
        attention, _ = self.mha(x, x, x, mask)
        attention = self.dropout1(attention, training=training)
        out1 = self.layernorm1(x + attention)

        feed_forward = self.dense_hidden(out1)
        feed_forward = self.dense_output(feed_forward)
        feed_forward = self.dropout2(feed_forward, training=training)
        return self.layernorm2(out1 + feed_forward)


class DecoderBlock(tf.keras.layers.Layer):
    """A Transformer decoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize decoder attention and feed-forward layers."""
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training,
             look_ahead_mask, padding_mask):
        """Run masked self-attention, cross-attention, and feed-forward."""
        attention1, _ = self.mha1(x, x, x, look_ahead_mask)
        attention1 = self.dropout1(attention1, training=training)
        out1 = self.layernorm1(x + attention1)

        attention2, _ = self.mha2(
            out1, encoder_output, encoder_output, padding_mask
        )
        attention2 = self.dropout2(attention2, training=training)
        out2 = self.layernorm2(out1 + attention2)

        feed_forward = self.dense_hidden(out2)
        feed_forward = self.dense_output(feed_forward)
        feed_forward = self.dropout3(feed_forward, training=training)
        return self.layernorm3(out2 + feed_forward)


class Encoder(tf.keras.layers.Layer):
    """The Transformer encoder stack."""

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """Initialize token embeddings, positions, and encoder blocks."""
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """Encode source token IDs into contextualized representations."""
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, x.dtype))
        x += tf.cast(self.positional_encoding[:seq_len], x.dtype)
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training=training, mask=mask)
        return x


class Decoder(tf.keras.layers.Layer):
    """The Transformer decoder stack."""

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """Initialize target embeddings, positions, and decoder blocks."""
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training,
             look_ahead_mask, padding_mask):
        """Decode target token IDs using encoder representations."""
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, x.dtype))
        x += tf.cast(self.positional_encoding[:seq_len], x.dtype)
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(
                x, encoder_output, training,
                look_ahead_mask, padding_mask
            )
        return x


class Transformer(tf.keras.Model):
    """Encoder-decoder Transformer for sequence-to-sequence translation."""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """Initialize the encoder, decoder, and vocabulary projection."""
        super().__init__()
        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_input, drop_rate
        )
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_target, drop_rate
        )
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """Return unnormalized target-vocabulary logits."""
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
