#!/usr/bin/env python3
"""Deep Neural Network module."""

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        """Initialize the deep neural network."""

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        for layer in layers:
            if not isinstance(layer, int) or layer <= 0:
                raise TypeError(
                    "layers must be a list of positive integers"
                )

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if i == 0:
                self.__weights['W{}'.format(i + 1)] = (
                    np.random.randn(layers[i], nx) *
                    np.sqrt(2 / nx)
                )
            else:
                self.__weights['W{}'.format(i + 1)] = (
                    np.random.randn(layers[i], layers[i - 1]) *
                    np.sqrt(2 / layers[i - 1])
                )

            self.__weights['b{}'.format(i + 1)] = (
                np.zeros((layers[i], 1))
            )

    @property
    def L(self):
        """Getter for number of layers."""
        return self.__L

    @property
    def cache(self):
        """Getter for cache."""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights."""
        return self.__weights

    def forward_prop(self, X):
        """Calculates forward propagation."""

        self.__cache['A0'] = X

        for i in range(self.__L):
            W = self.__weights['W{}'.format(i + 1)]
            b = self.__weights['b{}'.format(i + 1)]
            A_prev = self.__cache['A{}'.format(i)]

            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))

            self.__cache['A{}'.format(i + 1)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Calculates the cost."""

        m = Y.shape[1]

        cost = -np.sum(
            Y * np.log(A) +
            (1 - Y) * np.log(1.0000001 - A)
        ) / m

        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network."""

        A, _ = self.forward_prop(X)

        cost = self.cost(Y, A)

        prediction = np.where(A >= 0.5, 1, 0)

        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent."""

        m = Y.shape[1]
        weights_copy = self.__weights.copy()

        dZ = cache['A{}'.format(self.__L)] - Y

        for i in reversed(range(self.__L)):
            A_prev = cache['A{}'.format(i)]

            W = weights_copy['W{}'.format(i + 1)]

            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if i != 0:
                A = cache['A{}'.format(i)]
                dZ = np.matmul(W.T, dZ) * A * (1 - A)

            self.__weights['W{}'.format(i + 1)] = (
                self.__weights['W{}'.format(i + 1)] - alpha * dW
            )

            self.__weights['b{}'.format(i + 1)] = (
                self.__weights['b{}'.format(i + 1)] - alpha * db
            )

    def train(
        self,
        X,
        Y,
        iterations=5000,
        alpha=0.05,
        verbose=True,
        graph=True,
        step=100
    ):
        """Trains the deep neural network."""

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")

        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")

        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")

            if step <= 0 or step > iterations:
                raise ValueError(
                    "step must be positive and <= iterations"
                )

        costs = []
        steps = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)

            if i % step == 0 or i == iterations:
                current_cost = self.cost(Y, A)

                if verbose:
                    print(
                        "Cost after {} iterations: {}".format(
                            i,
                            current_cost
                        )
                    )

                costs.append(current_cost)
                steps.append(i)

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(steps, costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file."""

        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled object."""

        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
