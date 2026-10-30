# Transformer Applications

This project applies Transformer architectures to machine translation with
TensorFlow and Keras.

## General Learning Objectives

By completing this project, I should be able to:

- Use Transformer models for machine translation.
- Write custom training and testing loops in Keras.
- Load, transform, and process datasets with TensorFlow Datasets (TFDS).

## Project Workflow

The project builds a machine-translation pipeline in stages:

1. **Dataset preparation** — obtain the parallel text data needed for translation.
2. **Token encoding** — convert source and target sentences into token IDs.
3. **TensorFlow encoding** — represent encoded examples as TensorFlow-compatible data.
4. **Input pipeline** — prepare batched, shuffled, and prefetched training data.
5. **Mask creation** — create masks used by the Transformer to hide padding and future target tokens.
6. **Training and evaluation** — train the Transformer and evaluate its translation performance with custom Keras loops.

The exact function signatures, output contracts, and filenames for each numbered task
are defined by the corresponding task prompt. Implementations should follow those
requirements rather than relying on this overview alone.

## Key Concepts

### Transformer machine translation

A Transformer encoder processes the source sentence, while a decoder generates
the target sentence token by token. Attention lets the model relate relevant
source and target positions without using recurrent layers.

### Custom Keras training and testing loops

A custom training loop typically uses `tf.GradientTape` to calculate gradients,
updates model parameters with an optimizer, and tracks training metrics. A
testing loop runs inference without applying gradient updates and aggregates
evaluation metrics.

### TensorFlow Datasets

TFDS provides datasets as `tf.data.Dataset` pipelines. Dataset processing can
include mapping preprocessing functions, shuffling training examples, batching,
and prefetching to prepare input efficiently.

## Repository

- GitHub repository: `pilot-0077/holbertonschool-machine_learning`
- Directory: `supervised_learning/transformer_apps`

## Task 0: Dataset

File: `0-dataset.py`

### Purpose

Implement the `Dataset` class for Portuguese-to-English machine translation using the TED Talks parallel corpus.

### Dataset setup prerequisite

The original TFDS source archive is offline. Before running the task, follow the project instructions to:

1. Download and extract the dataset archive into `~/.cache/ted_hrlr/`.
2. The `setup.py` helper is included in this directory next to the numbered task files.
3. Install the required packages, including `transformers==4.44.2` and `tensorflow-datasets==4.9.2`, as specified by the project setup instructions.

The helper `load_pt2en(split)` from `setup.py` replaces the old `tfds.load('ted_hrlr_translate/pt_to_en', split=split, as_supervised=True)` call and returns a `tf.data.Dataset` of Portuguese/English string pairs. It reads the extracted files under `~/.cache/ted_hrlr/datasets/pt_to_en/`.

### Implementation

The constructor initializes:

- `data_train` with `load_pt2en('train')`.
- `data_valid` with `load_pt2en('validation')`.
- `tokenizer_pt` and `tokenizer_en` by calling `tokenize_dataset(data_train)`.

The `tokenize_dataset(data)` method:

1. Reads the Portuguese and English text pairs from `data`.
2. Decodes each byte string as UTF-8.
3. Loads `neuralmind/bert-base-portuguese-cased` for Portuguese and `bert-base-uncased` for English.
4. Trains new tokenizers from the training corpus with a vocabulary size of `2 ** 13`.
5. Returns the Portuguese and English tokenizers, in that order.

### Example

```python
Dataset = __import__('0-dataset').Dataset

data = Dataset()
print(type(data.data_train))
print(type(data.tokenizer_pt))
print(type(data.tokenizer_en))
```

## Task 1: Encode Tokens

File: `1-dataset.py`

### Task

Extend the `Dataset` class with `encode(pt, en)`, which converts a Portuguese sentence and its English translation into token ID lists.

The class loads the train and validation splits with `load_pt2en` and trains tokenizers on the training split:

- Portuguese: `neuralmind/bert-base-portuguese-cased`
- English: `bert-base-uncased`
- Maximum vocabulary size for each newly trained tokenizer: `2 ** 13`

### Encoding behavior

For each sentence, decode the `tf.Tensor` from UTF-8 bytes into text and tokenize it with `add_special_tokens=False`. Then add the task-specific boundary tokens:

- Start token: `tokenizer.vocab_size`
- End token: `tokenizer.vocab_size + 1`

The method returns `pt_tokens, en_tokens` as two Python lists. Portuguese and English use their own tokenizer vocabulary sizes.

### Example

```python
for pt, en in data.data_train.take(1):
    pt_tokens, en_tokens = data.encode(pt, en)
    print(pt_tokens)
    print(en_tokens)
```

## Task 2: TF Encode

File: `2-dataset.py`

### Task

Extend the `Dataset` class with `tf_encode(pt, en)`, a TensorFlow wrapper around the Python `encode` method, and map it over both the training and validation splits.

### TensorFlow wrapper

The method uses `tf.py_function` to call `self.encode` from a `tf.data.Dataset` pipeline:

- Inputs: `[pt, en]`
- Output types: `[tf.int64, tf.int64]`
- Dynamic shapes: call `set_shape([None])` on both returned tensors because tokenized sentence lengths vary.

It returns the Portuguese and English token tensors in that order.

### Dataset updates

After loading the raw training and validation splits and training the tokenizers on the training split, the constructor maps `self.tf_encode` over both datasets:

- `data_train = data_train.map(self.tf_encode)`
- `data_valid = data_valid.map(self.tf_encode)`

Each dataset then yields variable-length integer token tensors instead of raw text strings.

## Task 3: Pipeline

File: `3-dataset.py`

### Task

Extend the `Dataset` class with a complete `tf.data` input pipeline. The constructor takes `batch_size` and `max_len`.

### Training pipeline

1. Load the training split and encode sentence pairs with `tf_encode`.
2. Filter out any pair where the Portuguese or English token sequence is longer than `max_len`.
3. Cache the filtered dataset.
4. Shuffle it with `buffer_size=20000`.
5. Group examples into padded batches of `batch_size`, padding variable-length token sequences with zero.
6. Prefetch batches with `tf.data.experimental.AUTOTUNE`.

### Validation pipeline

1. Load and encode the validation split.
2. Apply the same maximum-length filter.
3. Group examples into padded batches of `batch_size`.

Validation is not shuffled, cached, or prefetched by this task specification.

### Length filtering

The filter keeps an example only when both conditions are true:

- `tf.size(pt) <= max_len`
- `tf.size(en) <= max_len`

The resulting batches have shapes `(batch_size, padded_pt_length)` and `(batch_size, padded_en_length)`; the padded lengths can differ between batches and between languages.

## Task 4: Create Masks

File: `4-create_masks.py`

### Task

Implement `create_masks(inputs, target)` using TensorFlow operations only. Padding token ID `0` is masked with values of `1.0`.

### Masks

- **Encoder mask**: identifies padding tokens in `inputs` and has shape `(batch_size, 1, 1, seq_len_in)`.
- **Look-ahead mask**: an upper-triangular matrix that masks future target positions while leaving the current and previous positions visible.
- **Target padding mask**: identifies padding tokens in `target`. The combined decoder mask is the element-wise maximum of this mask and the look-ahead mask, with shape `(batch_size, 1, seq_len_out, seq_len_out)`.
- **Decoder mask**: identifies padding positions in the source `inputs` for the decoder's second attention block; its shape is `(batch_size, 1, 1, seq_len_in)`.

### Return value

The function returns `encoder_mask, combined_mask, decoder_mask` in that order. Both encoder and decoder masks use the input padding positions; the combined mask uses target padding positions plus the look-ahead mask.

## Task 5: Train

Files: `5-transformer.py`, `5-train.py`

### Transformer model

`5-transformer.py` contains a self-contained encoder-decoder Transformer:

- Sinusoidal positional encoding.
- Scaled dot-product attention and multi-head attention.
- Encoder blocks with residual connections, layer normalization, and feed-forward layers.
- Decoder blocks with causal self-attention, encoder-decoder attention, residual connections, and feed-forward layers.
- Separate source/target embeddings and a final Dense projection that returns target-vocabulary logits.

The source and target vocabulary sizes are each the tokenizer vocabulary size plus two, to include the custom start and end token IDs.

### Training loop

`5-train.py` defines `train_transformer(N, dm, h, hidden, max_len, batch_size, epochs)`. It:

1. Creates the `Dataset(batch_size, max_len)` data pipeline.
2. Builds the Transformer with the Portuguese and English vocabulary sizes.
3. Uses Adam with `beta_1=0.9`, `beta_2=0.98`, and `epsilon=1e-9`.
4. Applies the Transformer learning-rate schedule with `warmup_steps=4000`:

   `dm ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)`

5. Uses teacher forcing: the decoder input is `target[:, :-1]` and the expected output is `target[:, 1:]`.
6. Computes sparse categorical cross-entropy from logits and excludes target padding tokens (ID `0`) from both loss and accuracy.
7. Prints running loss and accuracy every 50 batches and the final metrics at the end of each epoch.
8. Returns the trained Transformer model.

### Run

With the dataset archive, `setup.py`, and dependencies installed, run the supplied task test:

```bash
python3 5-main.py
```

A short two-epoch run is intended as a functional smoke test. Longer training (20 or more epochs) is needed for more substantial translation quality.

