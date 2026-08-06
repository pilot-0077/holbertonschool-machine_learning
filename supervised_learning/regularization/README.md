# Regularization

This project covers regularization techniques used to reduce overfitting and
improve the generalization of neural networks.

## Learning Objectives

- What regularization is and why it is used
- L1 and L2 regularization
- Dropout
- Early stopping
- Data augmentation
- Regularization with NumPy and TensorFlow
- Advantages and limitations of regularization methods

## Files

- `0-l2_reg_cost.py`: Calculates the cost of a neural network with L2
  regularization.

## L2 Regularization

L2 regularization adds a penalty based on the squared values of the network
weights:

```text
L2 cost = original cost + (lambda / (2 * m)) * sum(W^2)
