#!/usr/bin/env python3
"""Creates mini-batches for mini-batch gradient descent."""

shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """Creates shuffled mini-batches from input data and labels.

    Args:
        X: numpy.ndarray of shape (m, nx) containing input data.
        Y: numpy.ndarray of shape (m, ny) containing labels.
        batch_size: Number of data points in each batch.

    Returns:
        A list of tuples containing X and Y mini-batches.
    """
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    mini_batches = []
    m = X.shape[0]

    for start in range(0, m, batch_size):
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        Y_batch = Y_shuffled[start:end]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
