#!/usr/bin/env python3
"""Train and return a Gensim Word2Vec model."""

import inspect
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Create, build, and train a Word2Vec model."""
    word2vec = gensim.models.Word2Vec
    parameters = inspect.signature(word2vec).parameters

    # Gensim 4 renamed size -> vector_size and iter -> epochs.
    size_parameter = (
        "vector_size" if "vector_size" in parameters else "size"
    )
    epochs_parameter = "epochs" if "epochs" in parameters else "iter"

    model_args = {
        size_parameter: vector_size,
        "min_count": min_count,
        "window": window,
        "negative": negative,
        "sg": 0 if cbow else 1,
        epochs_parameter: epochs,
        "seed": seed,
        "workers": workers,
    }

    # Passing sentences to Word2Vec builds the vocabulary and trains it.
    return word2vec(sentences=sentences, **model_args)
