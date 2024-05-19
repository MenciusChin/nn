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
    e_x = np.exp(x - np.max(x))

    # Implement softmax division by dimension/axis
    return e_x / e_x.sum() if dim is None else e_x / e_x.sum(axis=dim)


#




if __name__ == "__main__":
    # test
    scores = np.array([1, 2, 3, 6])
    print(softmax(scores, dim=0))