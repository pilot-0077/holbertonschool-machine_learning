#!/usr/bin/env python3
"""Q-learning module."""

import numpy as np

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Train an agent using Q-learning.

    Args:
        env: FrozenLake environment.
        Q: Q-table.
        episodes: Number of training episodes.
        max_steps: Maximum steps per episode.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Initial epsilon value.
        min_epsilon: Minimum epsilon value.
        epsilon_decay: Epsilon decay rate.

    Returns:
        Q: Updated Q-table.
        total_rewards: Rewards obtained for each episode.
    """
    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)

            new_state, reward, terminated, truncated, _ = env.step(action)

            if terminated and reward == 0:
                reward = -1

            Q[state, action] = Q[state, action] + alpha * (
                reward
                + gamma * np.max(Q[new_state])
                - Q[state, action]
            )

            state = new_state
            episode_reward += reward

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)

          epsilon = min_epsilon + (1 - min_epsilon) * \
              np.exp(-epsilon_decay * episode)

    return Q, total_rewards
