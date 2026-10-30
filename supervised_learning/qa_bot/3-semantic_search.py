#!/usr/bin/env python3
"""Semantic search over a directory of reference documents."""

import os

import tensorflow as tf
import tensorflow_hub as hub


# Load Universal Sentence Encoder once and reuse it for all searches.
use_model = hub.load(
    "https://tfhub.dev/google/universal-sentence-encoder/4"
)


def semantic_search(corpus_path, sentence):
    """Return the document in corpus_path most similar to sentence."""
    if not isinstance(sentence, str) or not sentence.strip():
        return None

    document_paths = [
        os.path.join(corpus_path, filename)
        for filename in sorted(os.listdir(corpus_path))
        if os.path.isfile(os.path.join(corpus_path, filename))
        and not filename.startswith(".")
    ]

    if not document_paths:
        return None

    documents = []
    for document_path in document_paths:
        with open(document_path, encoding="utf-8") as file:
            document = file.read()
            if document.strip():
                documents.append(document)

    if not documents:
        return None

    # Encode the query and every document as sentence embeddings.
    query_embedding = use_model([sentence])
    document_embeddings = use_model(documents)

    # Compare normalized vectors with cosine similarity.
    query_embedding = tf.math.l2_normalize(query_embedding, axis=1)
    document_embeddings = tf.math.l2_normalize(
        document_embeddings,
        axis=1
    )
    similarities = tf.matmul(
        query_embedding,
        document_embeddings,
        transpose_b=True
    )

    best_index = int(tf.argmax(similarities[0]).numpy())
    return documents[best_index]
