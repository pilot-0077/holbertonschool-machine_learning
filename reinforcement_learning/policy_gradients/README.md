# Policy Gradients

This project introduces policy-based reinforcement learning methods, where an
agent learns a policy that directly maps states to action probabilities.

## Learning Objectives

- What a policy is in reinforcement learning
- How a policy can be represented with weights
- How to calculate a policy gradient
- How rewards influence policy updates
- What Monte-Carlo policy gradient methods are
- How the REINFORCE algorithm works

## Task 0 - Simple Policy Function

File: `policy_gradient.py`

The `policy(matrix, weight)` function computes action probabilities from a
state matrix and a weight matrix.

The calculation follows two steps:

1. Compute the policy logits with matrix multiplication.
2. Apply the softmax function to convert the logits into probabilities.

The returned probabilities sum to 1 for each state.

## Repository

- GitHub repository: `holbertonschool-machine_learning`
- Directory: `reinforcement_learning/policy_gradients`
