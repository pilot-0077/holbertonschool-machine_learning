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
