#!/usr/bin/env python3
"""Module that loads the FrozenLake environment."""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Load and return a FrozenLake environment.

    Args:
        desc: Custom description of the map.
        map_name: Name of a pre-made map.
        is_slippery: Whether the ice is slippery.

    Returns:
        The FrozenLake environment.
    """
    return gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode="ansi"
    )
