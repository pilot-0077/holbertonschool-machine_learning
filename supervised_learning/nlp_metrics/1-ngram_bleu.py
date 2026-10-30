#!/usr/bin/env python3
"""Calculate an n-gram BLEU score."""

from collections import Counter
import math


def ngram_bleu(references, sentence, n):
    """Calculate the n-gram BLEU score for a proposed sentence."""
    if not references or not sentence or n <= 0 or len(sentence) < n:
        return 0.0

    sentence_ngrams = Counter(
        tuple(sentence[i:i + n])
        for i in range(len(sentence) - n + 1)
    )

    max_reference_counts = {}
    for reference in references:
        reference_ngrams = Counter(
            tuple(reference[i:i + n])
            for i in range(len(reference) - n + 1)
        )
        for gram, count in reference_ngrams.items():
            max_reference_counts[gram] = max(
                max_reference_counts.get(gram, 0), count
            )

    clipped_count = sum(
        min(count, max_reference_counts.get(gram, 0))
        for gram, count in sentence_ngrams.items()
    )
    precision = clipped_count / sum(sentence_ngrams.values())

    if precision == 0:
        return 0.0

    candidate_length = len(sentence)
    reference_length = min(
        (len(reference) for reference in references),
        key=lambda length: (abs(length - candidate_length), length)
    )

    if candidate_length > reference_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(
            1 - reference_length / candidate_length
        )

    return brevity_penalty * precision
