#!/usr/bin/env python3
"""Calculate the unigram BLEU score."""

from collections import Counter
import math


def uni_bleu(references, sentence):
    """Calculate the unigram BLEU score for a proposed sentence."""
    if not references or not sentence:
        return 0.0

    sentence_counts = Counter(sentence)
    max_reference_counts = {}

    for reference in references:
        reference_counts = Counter(reference)
        for word, count in reference_counts.items():
            max_reference_counts[word] = max(
                max_reference_counts.get(word, 0), count
            )

    clipped_count = sum(
        min(count, max_reference_counts.get(word, 0))
        for word, count in sentence_counts.items()
    )
    precision = clipped_count / len(sentence)

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
