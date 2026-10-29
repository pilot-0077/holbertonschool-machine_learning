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

## Multi-Head Attention

File: `6-multihead_attention.py`

### Task

Implement the `MultiHeadAttention` class as a subclass of `tensorflow.keras.layers.Layer`.

The constructor accepts `dm` (model dimensionality) and `h` (number of attention heads), where `dm` must be divisible by `h`. Each head has depth `dm // h`.

The layer exposes `h`, `dm`, `depth`, `Wq`, `Wk`, `Wv`, and `linear`. The three first Dense layers project the inputs into query, key, and value representations; `linear` projects the concatenated attention output.

### Processing steps

1. Project `Q`, `K`, and `V` through their respective Dense layers.
2. Reshape each projection to split the model depth into `h` heads.
3. Transpose the tensors to shape `(batch, h, sequence_length, depth)`.
4. Call `sdp_attention` from `5-sdp_attention.py`.
5. Transpose and concatenate the heads, then apply the final Dense layer.

### Output shapes

- Output: `(batch, seq_len_q, dm)`
- Attention weights: `(batch, h, seq_len_q, seq_len_v)`

For the task's example, `MultiHeadAttention(512, 8)` has `depth = 64`.

## Transformer Encoder Block

File: `7-transformer_encoder_block.py`

### Task

Implement `EncoderBlock`, a subclass of `tensorflow.keras.layers.Layer`. The constructor takes `dm`, `h`, `hidden`, and `drop_rate=0.1`, and creates:

- `mha`: a `MultiHeadAttention` layer.
- `dense_hidden`: Dense layer with `hidden` units and ReLU activation.
- `dense_output`: Dense layer with `dm` units.
- `layernorm1` and `layernorm2`: LayerNormalization layers with `epsilon=1e-6`.
- `dropout1` and `dropout2`: Dropout layers using `drop_rate`.

### Forward pass

1. Apply multi-head self-attention to `x` as query, key, and value.
2. Apply dropout, add the original input (residual connection), then apply layer normalization.
3. Pass the result through the two feed-forward Dense layers.
4. Apply dropout, add the second residual connection, then apply the second layer normalization.

### Output shape

Input: `(batch, input_seq_len, dm)`

Output: `(batch, input_seq_len, dm)`
