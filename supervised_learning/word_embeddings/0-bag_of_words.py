#!/usr/bin/env python3
"""Create Bag of Words embeddings from a list of sentences."""

import re
import numpy as np


def bag_of_words(sentences, vocab=None):
    """Return a Bag of Words matrix and the feature vocabulary."""
    tokenized_sentences = []
    for sentence in sentences:
        # Treat possessives such as "children's" as "children".
        sentence = re.sub(r"['’]s\b", "", sentence.lower())
        tokenized_sentences.append(re.findall(r"[a-z0-9]+", sentence))

    if vocab is None:
        features = sorted({
            word
            for sentence in tokenized_sentences
            for word in sentence
        })
    else:
        # Keep the caller-provided vocabulary order.
        features = list(dict.fromkeys(
            re.sub(r"['’]s\b", "", word.lower()).strip()
            for word in vocab
        ))
        features = [word for word in features if word]

    word_to_index = {word: index for index, word in enumerate(features)}
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    for row, sentence in enumerate(tokenized_sentences):
        for word in sentence:
            index = word_to_index.get(word)
            if index is not None:
                embeddings[row, index] += 1

    return embeddings, features
