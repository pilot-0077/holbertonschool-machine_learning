# Temporal Difference

This project explores model-free reinforcement learning methods that learn
value estimates directly from experience.

## Learning Objectives

- Understand Monte Carlo methods
- Understand Temporal Difference learning
- Understand bootstrapping
- Understand n-step Temporal Difference
- Understand TD(lambda)
- Understand eligibility traces
- Understand SARSA, SARSA(lambda), and SARSAMAX
- Understand on-policy and off-policy learning

## Monte Carlo

Monte Carlo policy evaluation estimates the value of states from complete
episodes. After an episode ends, the return observed from each visited state is
used to update that state's value estimate.

Unlike Temporal Difference methods, Monte Carlo does not bootstrap from the
current estimate of a following state.

The incremental update is based on the difference between the observed return
and the current state-value estimate.

## Temporal Difference

Temporal Difference methods update value estimates before an episode has
finished. They combine observed rewards with estimated future values, which is
known as bootstrapping.

## Files

- `0-monte_carlo.py` - implements Monte Carlo policy evaluation

## Task 0 - Monte Carlo

The `monte_carlo` function receives an environment, an initial value estimate,
and a policy. It runs multiple episodes, computes discounted returns, and
updates the value estimate using the supplied learning rate and discount rate.

The implementation supports the Gymnasium API, including the `terminated` and
`truncated` episode-ending signals.
