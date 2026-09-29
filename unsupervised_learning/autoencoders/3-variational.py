#!/usr/bin/env python3
"""Variational autoencoder."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Create a variational autoencoder.

    Args:
        input_dims: Integer containing the dimensions of the model input.
        hidden_layers: List containing the number of nodes for each hidden
            layer in the encoder, respectively.
        latent_dims: Integer containing the dimensions of the latent space.

    Returns:
        encoder, decoder, auto
    """
    inputs = keras.Input(shape=(input_dims,))
    x = inputs

    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims)(x)
    z_log_var = keras.layers.Dense(latent_dims)(x)

    def sampling(args):
        """Sample from the latent distribution."""
        mean, log_var = args
        epsilon = keras.backend.random_normal(
            shape=keras.backend.shape(mean)
        )
        return mean + keras.backend.exp(log_var / 2) * epsilon

    z = keras.layers.Lambda(
        sampling,
        output_shape=(latent_dims,)
    )([z_mean, z_log_var])

    encoder = keras.Model(inputs, [z, z_mean, z_log_var])

    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    outputs = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(x)

    decoder = keras.Model(latent_inputs, outputs)

    decoded = decoder(z)
    auto = keras.Model(inputs, decoded)

    reconstruction_loss = keras.losses.binary_crossentropy(
        inputs,
        decoded
    )
    reconstruction_loss *= input_dims

    kl_loss = (
        1
        + z_log_var
        - keras.backend.square(z_mean)
        - keras.backend.exp(z_log_var)
    )
    kl_loss = -0.5 * keras.backend.sum(kl_loss, axis=-1)

    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam')

    return encoder, decoder, auto
