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

## Transformer Decoder Block

File: `8-transformer_decoder_block.py`

### Task

Implement `DecoderBlock`, a subclass of `tensorflow.keras.layers.Layer`, with two multi-head attention layers, a feed-forward network, three dropout layers, and three layer normalization layers.

### Forward pass

1. Apply masked self-attention to `x` using `look_ahead_mask`.
2. Apply `dropout1`, add the residual input `x`, and normalize with `layernorm1`.
3. Apply encoder-decoder attention using the first sublayer output as query and `encoder_output` as key and value, with `padding_mask`.
4. Apply `dropout2`, add the residual connection, and normalize with `layernorm2`.
5. Pass the result through `dense_hidden` (ReLU) and `dense_output`.
6. Apply `dropout3`, add the second residual connection, and normalize with `layernorm3`.

### Output shape

Input target: `(batch, target_seq_len, dm)`

Output: `(batch, target_seq_len, dm)`

## Transformer Encoder

File: `9-transformer_encoder.py`

### Task

Implement `Encoder`, a subclass of `tensorflow.keras.layers.Layer`, with:

- `N`: number of encoder blocks.
- `dm`: model dimensionality.
- `embedding`: token embedding layer with `input_vocab` entries and `dm` dimensions.
- `positional_encoding`: NumPy positional encoding matrix of shape `(max_seq_len, dm)`.
- `blocks`: a list containing `N` independently constructed `EncoderBlock` instances.
- `dropout`: dropout layer applied after adding positional encodings.

### Forward pass

1. Embed the input token IDs.
2. Scale embeddings by `sqrt(dm)`.
3. Add the positional encodings up to the input sequence length.
4. Apply dropout.
5. Pass the sequence through each encoder block in order, forwarding `training` and `mask`.
6. Return the encoded tensor.

### Output shape

Input token IDs: `(batch, input_seq_len)`

Output: `(batch, input_seq_len, dm)`

Each entry in `blocks` must be a distinct layer instance; do not create the list by multiplying a single layer instance.

## Transformer Decoder

File: `10-transformer_decoder.py`

### Task

Implement `Decoder`, a subclass of `tensorflow.keras.layers.Layer`, with:

- `N`: number of decoder blocks.
- `dm`: model dimensionality.
- `embedding`: target token embedding layer.
- `positional_encoding`: NumPy array with shape `(max_seq_len, dm)`.
- `blocks`: a list of `N` independently constructed `DecoderBlock` instances.
- `dropout`: dropout layer applied after positional encodings are added.

### Forward pass

1. Embed target token IDs.
2. Scale the embeddings by `sqrt(dm)`.
3. Add positional encodings up to the target sequence length.
4. Apply dropout.
5. Pass the sequence through every decoder block, forwarding the encoder output, training flag, look-ahead mask, and padding mask.
6. Return the decoder output.

### Output shape

Input target token IDs: `(batch, target_seq_len)`

Output: `(batch, target_seq_len, dm)`

Each element of `blocks` must be a distinct `DecoderBlock` instance.

## Transformer Network

File: `11-transformer.py`

### Task

Implement `Transformer`, a subclass of `tensorflow.keras.Model`, with an encoder, decoder, and final Dense layer:

- `encoder`: an `Encoder` configured with `N`, `dm`, `h`, `hidden`, `input_vocab`, and `max_seq_input`.
- `decoder`: a `Decoder` configured with `N`, `dm`, `h`, `hidden`, `target_vocab`, and `max_seq_target`.
- `linear`: a Dense layer with `target_vocab` output units.

### Forward pass

1. Encode the source token IDs using `encoder_mask`.
2. Pass target token IDs and the encoder output to the decoder, along with `look_ahead_mask` and `decoder_mask`.
3. Apply the final Dense layer to decoder outputs to produce target-vocabulary logits.

### Output shape

Input source: `(batch, input_seq_len)`

Input target: `(batch, target_seq_len)`

Output: `(batch, target_seq_len, target_vocab)`

# RNN Encoder

File: `0-rnn_encoder.py`

## Task

Implement `RNNEncoder`, a subclass of `tensorflow.keras.layers.Layer`, to encode source token sequences for machine translation.

The constructor `RNNEncoder(vocab, embedding, units, batch)` creates these public attributes:

- `batch`: batch size.
- `units`: number of GRU hidden units.
- `embedding`: a Keras Embedding layer with `vocab` entries and embedding dimension `embedding`.
- `gru`: a Keras GRU layer with `units` units, `recurrent_initializer='glorot_uniform'`, `return_sequences=True`, and `return_state=True`.

## Methods

### `initialize_hidden_state()`

Returns a zero tensor of shape `(batch, units)`.

### `call(x, initial)`

1. Convert input token IDs into embedding vectors.
2. Pass the embeddings and `initial` hidden state to the GRU.
3. Return `outputs, hidden`.

## Tensor shapes

- Input token IDs: `(batch, input_seq_len)`
- Initial hidden state: `(batch, units)`
- Outputs: `(batch, input_seq_len, units)`
- Final hidden state: `(batch, units)`

## Example

```python
encoder = RNNEncoder(1024, 128, 256, 32)
initial = encoder.initialize_hidden_state()
outputs, hidden = encoder(x, initial)
```
