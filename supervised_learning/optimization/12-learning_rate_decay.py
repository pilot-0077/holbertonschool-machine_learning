#!/usr/bin/env python3
"""Creates an inverse time learning rate decay schedule."""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """Creates an inverse time learning rate decay schedule.

    Args:
        alpha: Initial learning rate.
        decay_rate: Rate at which the learning rate decays.
        decay_step: Number of steps before each decay.

    Returns:
        A TensorFlow inverse time decay schedule.
    """
    schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )

    return schedule
