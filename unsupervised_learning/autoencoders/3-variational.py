#!/usr/bin/env python3
"""Variational autoencoder implementation."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Create a variational autoencoder.

    Args:
        input_dims: Integer containing the dimensions of the model input.
        hidden_layers: List containing the number of nodes for each hidden
            layer in the encoder. The order is reversed for the decoder.
        latent_dims: Integer containing the dimensions of the latent space.

    Returns:
        encoder: Model that outputs the latent representation, mean, and
            log variance, respectively.
        decoder: Decoder model.
        auto: Full variational autoencoder model.
    """
    encoder_input = keras.Input(shape=(input_dims,))
    encoded = encoder_input

    for units in hidden_layers:
        encoded = keras.layers.Dense(
            units=units,
            activation='relu'
        )(encoded)

    z_mean = keras.layers.Dense(
        units=latent_dims,
        activation=None
    )(encoded)
    z_log_var = keras.layers.Dense(
        units=latent_dims,
        activation=None
    )(encoded)

    def sampling(args):
        """Sample a latent vector using the reparameterization trick."""
        mean, log_var = args
        epsilon = keras.backend.random_normal(
            shape=keras.backend.shape(mean)
        )
        return mean + keras.backend.exp(log_var / 2) * epsilon

    z = keras.layers.Lambda(
        sampling,
        output_shape=(latent_dims,)
    )([z_mean, z_log_var])

    encoder = keras.Model(
        inputs=encoder_input,
        outputs=[z, z_mean, z_log_var]
    )

    decoder_input = keras.Input(shape=(latent_dims,))
    decoded = decoder_input

    for units in reversed(hidden_layers):
        decoded = keras.layers.Dense(
            units=units,
            activation='relu'
        )(decoded)

    decoder_output = keras.layers.Dense(
        units=input_dims,
        activation='sigmoid'
    )(decoded)

    decoder = keras.Model(
        inputs=decoder_input,
        outputs=decoder_output
    )

    latent_output = encoder(encoder_input)[0]
    reconstructed = decoder(latent_output)
    auto = keras.Model(
        inputs=encoder_input,
        outputs=reconstructed
    )

    def vae_loss(inputs, outputs):
        """Calculate reconstruction and KL-divergence loss."""
        reconstruction = keras.backend.binary_crossentropy(
            inputs,
            outputs
        )
        reconstruction = keras.backend.sum(reconstruction, axis=1)

        kl_divergence = 1 + z_log_var
        kl_divergence -= keras.backend.square(z_mean)
        kl_divergence -= keras.backend.exp(z_log_var)
        kl_divergence = keras.backend.sum(kl_divergence, axis=1)
        kl_divergence *= -0.5

        return reconstruction + kl_divergence

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
