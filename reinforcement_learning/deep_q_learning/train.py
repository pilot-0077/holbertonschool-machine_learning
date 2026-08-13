#!/usr/bin/env python3
"""Train a Deep Q-Network agent to play Atari Breakout."""

import importlib

import gymnasium as gym
import keras
import numpy as np
from PIL import Image


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4
TRAINING_STEPS = 1000000


def patch_keras_rl():
    """Expose the Keras version expected by keras-rl2."""
    tf_keras = importlib.import_module("tensorflow.keras")

    if not hasattr(tf_keras, "__version__"):
        tf_keras.__version__ = keras.__version__


patch_keras_rl()

dqn_module = importlib.import_module("rl.agents.dqn")
DQNAgent = dqn_module.DQNAgent

core_module = importlib.import_module("rl.core")
Processor = core_module.Processor

memory_module = importlib.import_module("rl.memory")
SequentialMemory = memory_module.SequentialMemory

policy_module = importlib.import_module("rl.policy")
EpsGreedyQPolicy = policy_module.EpsGreedyQPolicy
LinearAnnealedPolicy = policy_module.LinearAnnealedPolicy

layers_module = importlib.import_module("tensorflow.keras.layers")
Activation = layers_module.Activation
Conv2D = layers_module.Conv2D
Dense = layers_module.Dense
Flatten = layers_module.Flatten
Permute = layers_module.Permute

models_module = importlib.import_module("tensorflow.keras.models")
Sequential = models_module.Sequential

optimizer_module = importlib.import_module(
    "tensorflow.keras.optimizers.legacy"
)
Adam = optimizer_module.Adam


class KerasRLWrapper(gym.Wrapper):
    """Adapt Gymnasium to the API expected by keras-rl2."""

    def reset(self, **kwargs):
        """Reset the environment and return the observation."""
        observation, _ = self.env.reset(**kwargs)
        return observation

    def step(self, action):
        """Perform an action using the old four-value step API."""
        observation, reward, terminated, truncated, info = \
            self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, info

    def render(self, mode="human"):
        """Render the wrapped environment."""
        return self.env.render()


class AtariProcessor(Processor):
    """Preprocess Atari observations and rewards."""

    def process_observation(self, observation):
        """Resize a frame and convert it to grayscale."""
        image = Image.fromarray(observation)
        image = image.resize(INPUT_SHAPE).convert("L")
        return np.array(image).astype("uint8")

    def process_state_batch(self, batch):
        """Normalize a batch of observations."""
        return batch.astype("float32") / 255.0

    def process_reward(self, reward):
        """Clip a reward to the range minus one to one."""
        return np.clip(reward, -1.0, 1.0)


def build_model(nb_actions):
    """Build the convolutional policy network."""
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE

    model = Sequential()

    model.add(
        Permute(
            (2, 3, 1),
            input_shape=input_shape
        )
    )

    model.add(
        Conv2D(
            32,
            (8, 8),
            strides=(4, 4)
        )
    )
    model.add(Activation("relu"))

    model.add(
        Conv2D(
            64,
            (4, 4),
            strides=(2, 2)
        )
    )
    model.add(Activation("relu"))

    model.add(
        Conv2D(
            64,
            (3, 3),
            strides=(1, 1)
        )
    )
    model.add(Activation("relu"))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dense(nb_actions))
    model.add(Activation("linear"))

    return model


def build_agent(model, nb_actions, processor):
    """Build and compile the DQN agent."""
    memory = SequentialMemory(
        limit=1000000,
        window_length=WINDOW_LENGTH
    )

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.1,
        value_test=0.05,
        nb_steps=TRAINING_STEPS
    )

    agent = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        policy=policy,
        processor=processor,
        nb_steps_warmup=50000,
        gamma=0.99,
        target_model_update=10000,
        train_interval=4,
        delta_clip=1.0
    )

    agent.compile(
        Adam(learning_rate=0.00025),
        metrics=["mae"]
    )

    return agent


def main():
    """Train the Breakout agent and save its policy."""
    np.random.seed(123)

    base_env = gym.make("ALE/Breakout-v5")
    env = KerasRLWrapper(base_env)

    env.reset(seed=123)

    nb_actions = env.action_space.n
    model = build_model(nb_actions)
    processor = AtariProcessor()
    agent = build_agent(model, nb_actions, processor)

    model.summary()

    agent.fit(
        env,
        nb_steps=TRAINING_STEPS,
        visualize=False,
        verbose=1,
        log_interval=10000
    )

    agent.save_weights("policy.h5", overwrite=True)

    env.close()


if __name__ == "__main__":
    main()
