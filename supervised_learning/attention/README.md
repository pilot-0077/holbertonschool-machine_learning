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
