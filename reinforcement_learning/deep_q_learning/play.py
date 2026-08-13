#!/usr/bin/env python3
"""Play Atari Breakout using a trained Deep Q-Network policy."""

import importlib

import gymnasium as gym

import train


GreedyQPolicy = importlib.import_module(
    "rl.policy"
).GreedyQPolicy


def build_play_agent(model, nb_actions, processor):
    """Build and compile a greedy DQN agent for evaluation."""
    memory = train.SequentialMemory(
        limit=1000000,
        window_length=train.WINDOW_LENGTH
    )

    policy = GreedyQPolicy()

    agent = train.DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        policy=policy,
        test_policy=policy,
        processor=processor,
        nb_steps_warmup=50000,
        gamma=0.99,
        target_model_update=10000,
        train_interval=4,
        delta_clip=1.0
    )

    agent.compile(
        train.Adam(learning_rate=0.00025),
        metrics=["mae"]
    )

    return agent


def main():
    """Load the trained policy and display one Breakout episode."""
    base_env = gym.make(
        "ALE/Breakout-v5",
        render_mode="human"
    )
    env = train.KerasRLWrapper(base_env)

    nb_actions = env.action_space.n

    model = train.build_model(nb_actions)
    processor = train.AtariProcessor()

    agent = build_play_agent(
        model,
        nb_actions,
        processor
    )

    agent.load_weights("policy.h5")

    agent.test(
        env,
        nb_episodes=1,
        visualize=True
    )

    env.close()


if __name__ == "__main__":
    main()
