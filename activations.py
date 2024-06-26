"""
Activation functions and its derivatives.
"""

### Tasks ###
# 1. Add softmax division by dimension (2+)


import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1-np.tanh(x)**2


def softmax(x, dim=None):
    # Now only implemented the 1D array calculation
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))

    # Implement softmax division by dimension/axis
    return e_x / np.sum(e_x, axis=dim, keepdims=True)


def log_softmax(x, dim=None):
    max_x = np.max(x, axis=dim, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(x - max_x), axis=dim, keepdims=True)) + max_x
    return x - log_sum_exp


if __name__ == "__main__":
    # test
    scores = np.array([1, 2, 3, 6])
    print(softmax(scores, dim=0))
    print(log_softmax(scores, dim=0))