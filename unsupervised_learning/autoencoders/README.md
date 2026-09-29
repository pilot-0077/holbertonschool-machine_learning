# Autoencoders

This project covers autoencoders and their use for learning compact latent
representations of data.

## Learning Objectives

By the end of this project, I should be able to explain:

- What an autoencoder is
- What latent space is
- What a bottleneck is
- What a sparse autoencoder is
- What a convolutional autoencoder is
- What a generative model is
- What a variational autoencoder (VAE) is
- What Kullback-Leibler divergence is and how it is used in VAEs

## Task 0 - Vanilla Autoencoder

File: `0-vanilla.py`

The function:

```python
def autoencoder(input_dims, hidden_layers, latent_dims):
```

creates three Keras models:

- `encoder`: maps the input to a latent representation
- `decoder`: reconstructs the input from the latent representation
- `auto`: combines the encoder and decoder into the full autoencoder

The encoder uses the sizes in `hidden_layers` in order, while the decoder uses
them in reverse order. Hidden and latent layers use ReLU activation, and the
final decoder layer uses sigmoid activation.

The full autoencoder is compiled with the Adam optimizer and binary
cross-entropy loss.

## Repository

- GitHub repository: `holbertonschool-machine_learning`
- Directory: `unsupervised_learning/autoencoders`
