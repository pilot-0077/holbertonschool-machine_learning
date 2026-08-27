#!/usr/bin/env python3
"""Monte Carlo policy evaluation for reinforcement learning."""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """Perform Monte Carlo policy evaluation and update the value estimate.

    Args:
        env: Environment instance.
        V: NumPy array of shape (s,) containing state-value estimates.
        policy: Function taking a state and returning the next action.
        episodes: Number of episodes used for training.
        max_steps: Maximum number of steps allowed in each episode.
        alpha: Learning rate.
        gamma: Discount factor.

    Returns:
        The updated state-value estimate V.
    """
    for episode in range(episodes):
        state, _ = env.reset()
        episode_data = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_data.append((state, reward))

            if terminated or truncated:
                break

            state = next_state

        total_return = 0
        episode_data = np.array(episode_data, dtype=int)

        for state, reward in reversed(episode_data):
            total_return = reward + gamma * total_return

            if state not in episode_data[:episode, 0]:
                V[state] += alpha * (total_return - V[state])

    return V
