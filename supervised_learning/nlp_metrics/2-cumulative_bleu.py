#!/usr/bin/env python3
"""Calculate a cumulative n-gram BLEU score."""

from collections import Counter
import math


def cumulative_bleu(references, sentence, n):
    """Calculate the cumulative n-gram BLEU score for a proposed sentence."""
    if not references or not sentence or n <= 0 or len(sentence) < n:
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

    log_precisions = []

    for order in range(1, n + 1):
        candidate_ngrams = Counter(
            tuple(sentence[i:i + order])
            for i in range(candidate_length - order + 1)
        )

        max_reference_counts = {}
        for reference in references:
            reference_ngrams = Counter(
                tuple(reference[i:i + order])
                for i in range(len(reference) - order + 1)
            )
            for gram, count in reference_ngrams.items():
                max_reference_counts[gram] = max(
                    max_reference_counts.get(gram, 0), count
                )

        clipped_count = sum(
            min(count, max_reference_counts.get(gram, 0))
            for gram, count in candidate_ngrams.items()
        )
        total_count = sum(candidate_ngrams.values())
        precision = clipped_count / total_count if total_count else 0.0

        if precision == 0:
            return 0.0

        log_precisions.append(math.log(precision))

    # Equal weights for each n-gram order: geometric mean of precisions.
    geometric_mean = math.exp(sum(log_precisions) / n)
    return brevity_penalty * geometric_mean
