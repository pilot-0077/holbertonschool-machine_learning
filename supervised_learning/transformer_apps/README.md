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
