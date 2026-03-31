#!/usr/bin/env python3
"""
Module that contains the function create_confusion_matrix.
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Args:
        labels: a one-hot numpy.ndarray of shape (m, classes)
                containing the correct labels
        logits: a one-hot numpy.ndarray of shape (m, classes)
                containing the predicted labels

    Returns:
        A confusion numpy.ndarray of shape (classes, classes)
        with row indices representing the correct labels and
        column indices representing the predicted labels
    """
    true_labels = np.argmax(labels, axis=1)
    predicted_labels = np.argmax(logits, axis=1)
    classes = labels.shape[1]
    confusion = np.zeros((classes, classes))

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        confusion[true_label, predicted_label] += 1

    return confusion
