# Deep Q-Learning

This project implements a Deep Q-Network (DQN) agent that learns to play
Atari Breakout using Keras, keras-rl2, and Gymnasium.

## Learning Objectives

- Understand Deep Q-learning
- Understand policy networks
- Understand replay memory
- Understand target networks
- Understand why DQN uses two separate networks
- Use keras-rl2 to build and train a DQN agent

## Environment

The agent is trained using the Atari Breakout environment:

`ALE/Breakout-v5`

The original RGB observations are resized to 84x84 pixels and converted
to grayscale before being passed to the neural network.

Four consecutive frames are used as the state representation.

## DQN Architecture

The policy network contains:

- Convolutional layer with 32 filters
- Convolutional layer with 64 filters
- Convolutional layer with 64 filters
- Fully connected layer with 512 units
- Output layer containing one Q-value for each available action

## Replay Memory

A `SequentialMemory` replay buffer stores previous experiences and allows
the agent to learn from randomly sampled transitions.

## Policy

Training uses an epsilon-greedy policy with decreasing exploration.

Gameplay uses `GreedyQPolicy`, which always selects the action with the
highest predicted Q-value.

## Files

- `train.py` - builds and trains the DQN agent and saves its weights as
  `policy.h5`
- `play.py` - loads `policy.h5` and displays a game played by the trained
  agent

## Requirements

- Python 3.9
- NumPy 1.25.2
- Gymnasium 0.29.1
- TensorFlow 2.15.0
- Keras 2.15.0
- keras-rl2 1.0.4
- Pillow 10.3.0
- h5py 3.11.0
