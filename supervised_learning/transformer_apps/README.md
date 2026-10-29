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

