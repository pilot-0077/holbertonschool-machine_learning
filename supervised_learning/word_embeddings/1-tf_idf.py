#!/usr/bin/env python3
"""Create TF-IDF embeddings from a list of sentences."""

import re
import numpy as np


def tf_idf(sentences, vocab=None):
    """Return a TF-IDF matrix and the feature vocabulary."""
    tokenized_sentences = []
    for sentence in sentences:
        sentence = re.sub(r"['’]s\b", "", sentence.lower())
        tokenized_sentences.append(re.findall(r"(?u)\b\w\w+\b", sentence))

    if vocab is None:
        features = sorted({
            word
            for sentence in tokenized_sentences
            for word in sentence
        })
    else:
        features = list(dict.fromkeys(
            re.sub(r"['’]s\b", "", word.lower()).strip()
            for word in vocab
        ))
        features = [word for word in features if word]

    word_to_index = {word: index for index, word in enumerate(features)}
    n_documents = len(sentences)
    n_features = len(features)
    term_frequency = np.zeros((n_documents, n_features), dtype=float)

    for row, sentence in enumerate(tokenized_sentences):
        for word in sentence:
            index = word_to_index.get(word)
            if index is not None:
                term_frequency[row, index] += 1.0

    if n_documents == 0 or n_features == 0:
        return term_frequency, np.array(features)

    # Smooth IDF, matching the standard sklearn TfidfVectorizer formula.
    document_frequency = np.count_nonzero(term_frequency, axis=0)
    inverse_document_frequency = (
        np.log((1 + n_documents) / (1 + document_frequency)) + 1
    )
    embeddings = term_frequency * inverse_document_frequency

    # L2-normalize each sentence vector.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    return embeddings, np.array(features)
