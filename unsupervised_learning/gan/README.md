# Generative Adversarial Networks

This project covers Generative Adversarial Networks (GANs), including the
standard GAN training process and Wasserstein-based variants.

## Learning Objectives

By the end of this project, I should be able to explain:

- What a Generative Adversarial Network is
- The roles of the generator and discriminator
- How adversarial training works
- What a latent vector is
- What latent space is
- How a generator transforms latent vectors into synthetic samples
- What Wasserstein GANs are
- Why a critic is used in Wasserstein GANs
- Common GAN training issues such as instability and mode collapse

## Task 0 - Simple GAN

File: `0-simple_gan.py`

The `Simple_GAN` class inherits from `keras.Model` and manages a generator,
a discriminator, a latent-vector generator, and a collection of real examples.

The implementation uses a least-squares objective:

- The discriminator learns to score real examples as `1` and fake examples
  as `-1`.
- The generator learns to produce fake examples that the discriminator scores
  as `1`.
- The discriminator is updated `disc_iter` times for each generator update.
- Both networks use Adam optimizers with configurable learning rate and the
  GAN-specific beta values used by the task.

The class provides:

- `get_fake_sample()` to generate samples from latent vectors
- `get_real_sample()` to randomly select real training samples
- `train_step()` to perform discriminator and generator gradient updates

## Repository

- GitHub repository: `holbertonschool-machine_learning`
- Directory: `unsupervised_learning/gan`
