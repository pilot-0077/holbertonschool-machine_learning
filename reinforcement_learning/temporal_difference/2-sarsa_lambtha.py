#!/usr/bin/env python3
"""SARSA(lambda) algorithm with eligibility traces."""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Select an action using an epsilon-greedy policy.

    Args:
        Q: NumPy array containing the Q-table.
        state: Current environment state.
        epsilon: Probability of selecting a random action.

    Returns:
        The action to take.
    """
    if np.random.uniform(0, 1) > epsilon:
        return np.argmax(Q[state, :])

    return np.random.randint(0, Q.shape[1])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """Perform SARSA(lambda) using accumulating eligibility traces.

    Args:
        env: Environment instance.
        Q: NumPy array of shape (s, a) containing the Q-table.
        lambtha: Eligibility trace factor.
        episodes: Total number of episodes to train over.
        max_steps: Maximum number of steps per episode.
        alpha: Learning rate.
        gamma: Discount rate.
        epsilon: Initial epsilon-greedy exploration threshold.
        min_epsilon: Minimum value to which epsilon can decay.
        epsilon_decay: Decay rate applied to epsilon between episodes.

    Returns:
        The updated Q-table.
    """
    initial_epsilon = epsilon

    for episode in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        eligibility = np.zeros_like(Q)

        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            delta = (reward + gamma * Q[next_state, next_action] -
                     Q[state, action])

            eligibility *= gamma * lambtha
            eligibility[state, action] += 1
            Q += alpha * delta * eligibility

            if terminated or truncated:
                break

            state = next_state
            action = next_action

        epsilon = (min_epsilon + (initial_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * episode))

    return Q
