#!/usr/bin/env python3
"""Module for loading the FrozenLake environment."""

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Load a FrozenLake environment.

    Args:
        desc: Custom map description.
        map_name: Name of a pre-made map.
        is_slippery: Whether the ice is slippery.

    Returns:
        The FrozenLake environment.
    """
    if desc is None and map_name is None:
        desc = generate_random_map(size=8)

    return gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
    )
