#!/usr/bin/env python3
"""Convert a trained Gensim Word2Vec model to a trainable Keras Embedding."""

import tensorflow as tf


def gensim_to_keras(model):
    """Return a trainable Keras Embedding initialized with Word2Vec weights."""
    weights = model.wv.vectors
    embedding = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True,
    )
    return embedding
