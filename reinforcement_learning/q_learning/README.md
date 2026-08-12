# Q-Learning

This directory contains tasks related to reinforcement learning and Q-learning.

## Learning Objectives

By the end of this project, I should be able to explain:

- What a Markov Decision Process is
- What an environment is
- What an agent is
- What a state is
- What a policy function is
- What a value function is
- What a state-value function is
- What an action-value function is
- What a discount factor is
- What the Bellman equation is
- What epsilon-greedy is
- What Q-learning is

## Environment

The project uses Gymnasium and the FrozenLake environment.

Required packages:

```bash
pip install --user gymnasium==0.29.1
pip install --user Pillow==10.3.0
pip install --user h5py==3.11.0
```

## Tasks

### 0. Load the Environment

`0-load_env.py` contains the function:

```python
def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
```

The function loads the Gymnasium `FrozenLake-v1` environment.

If both `desc` and `map_name` are `None`, a random 8x8 map is generated.
