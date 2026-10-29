# Attention

This directory contains exercises on attention mechanisms and Transformer models.

## Positional Encoding

File: `4-positional_encoding.py`

### Task

Implement:

```python
def positional_encoding(max_seq_len, dm):
```

The function returns a NumPy array of shape `(max_seq_len, dm)` containing sinusoidal positional encodings.

### How it works

Transformers do not inherently represent token order, so positional encodings are added to token embeddings to provide position information.

For position \(pos\) and dimension \(i\), the standard sinusoidal encoding is:

\[
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/dm}}\right)
\]

\[
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/dm}}\right)
\]

Even dimensions use sine and odd dimensions use cosine.

### Example

```python
#!/usr/bin/env python3
import numpy as np

positional_encoding = __import__(
    '4-positional_encoding'
).positional_encoding

PE = positional_encoding(30, 512)
print(PE.shape)
```

Expected output:

```text
(30, 512)
```

### Requirements

- Use `import numpy as np`.
- Return a `numpy.ndarray` with shape `(max_seq_len, dm)`.
- The first position (`pos = 0`) has sine values of 0 and cosine values of 1.

## Scaled Dot Product Attention

File: `5-sdp_attention.py`

### Task

Implement:

```python
def sdp_attention(Q, K, V, mask=None):
```

The function computes scaled dot product attention and returns `output, weights`.

### Formula

```text
scores = Q @ Kᵀ / sqrt(dk)
weights = softmax(scores)
output = weights @ V
```

Here, `dk` is the depth of the query/key vectors. Dividing by `sqrt(dk)` keeps the dot-product scores at a suitable scale before applying softmax.

If `mask` is supplied, it is cast to the score tensor's data type and added as `mask * -1e9` before softmax. This convention assumes masked positions are marked with 1 and unmasked positions with 0.

### Shapes

- `Q`: `(..., seq_len_q, dk)`
- `K`: `(..., seq_len_v, dk)`
- `V`: `(..., seq_len_v, dv)`
- `weights`: `(..., seq_len_q, seq_len_v)`
- `output`: `(..., seq_len_q, dv)`

The softmax is applied over the final axis so each query's attention weights sum to 1 across the key/value sequence.
