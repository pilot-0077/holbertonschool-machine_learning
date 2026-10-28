#!/usr/bin/env python3
"""Train and return a Gensim Word2Vec model."""

import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Create, build, and train a Word2Vec model."""
    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=0 if cbow else 1,
        epochs=epochs,
        seed=seed,
        workers=workers,
    )
    return model
