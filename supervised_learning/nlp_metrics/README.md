# NLP Metrics

This directory contains exercises for evaluating natural language processing models.

## 0. Unigram BLEU score

**File:** `0-uni_bleu.py`

Implements `uni_bleu(references, sentence)`, which calculates unigram BLEU for a proposed sentence against one or more reference translations.

The implementation counts candidate unigrams, clips each count to the maximum count found in any reference, calculates unigram precision, selects the closest reference length (choosing the shorter length in a tie), and applies the brevity penalty.

Formula:

`BLEU-1 = brevity_penalty * clipped_unigram_precision`

For candidate length `c` and selected reference length `r`:

- `BP = 1` when `c > r`
- `BP = exp(1 - r / c)` when `c <= r`

Example:

```python
references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]
print(uni_bleu(references, sentence))
# 0.6549846024623855
```

## 1. N-gram BLEU score

**File:** `1-ngram_bleu.py`

Implements `ngram_bleu(references, sentence, n)`, calculating modified precision for candidate n-grams of size `n`, clipping counts to the maximum count in any reference, and applying the BLEU brevity penalty using the closest reference length (shorter reference wins a tie).

Formula: `N-gram BLEU = brevity_penalty * clipped_n_gram_precision`.

Example:

```python
references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]
print(ngram_bleu(references, sentence, 2))
# 0.6140480648084865
```

## 2. Cumulative N-gram BLEU score

**File:** `2-cumulative_bleu.py`

Implements `cumulative_bleu(references, sentence, n)`. It calculates clipped modified precision for every n-gram order from 1 through `n`, combines those precisions using an equally weighted geometric mean, and applies the brevity penalty once.

Formula:

`Cumulative BLEU = BP * exp((log(p1) + ... + log(pn)) / n)`

If any n-gram precision is zero, the cumulative score is zero.

Example:

```python
references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]
print(cumulative_bleu(references, sentence, 4))
# 0.5475182535069453
```
