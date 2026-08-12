#!/usr/bin/env python3
"""Module for playing an episode using a trained Q-table."""

import numpy as np


def play(env, Q, max_steps=100):
    """Play an episode using the trained agent.

    Args:
        env: FrozenLake environment.
        Q: Q-table containing action values.
        max_steps: Maximum number of steps in the episode.

    Returns:
        total_rewards: Total reward obtained during the episode.
        rendered_outputs: List of rendered board states.
    """
    state, _ = env.reset()
    total_rewards = 0
    rendered_outputs = [env.render()]

    for _ in range(max_steps):
        action = np.argmax(Q[state, :])

        new_state, reward, terminated, truncated, _ = env.step(action)

        total_rewards += reward
        rendered_outputs.append(env.render())

        state = new_state

        if terminated or truncated:
            break

    return total_rewards, rendered_outputs
