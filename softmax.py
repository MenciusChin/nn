"""
Softmax activation Layer.
"""

import numpy as np
import functional as F
from activation_layer import ActivationLayer
from activations import softmax


class Softmax(ActivationLayer):
    def __init__(self, activation=softmax):
        self.activation = softmax

    
    def forward(self, input, dim):
        """
        Expected input shape:
        (N, num_classes)
        """
        self.input = input
        self.softmax = softmax(input, dim)

        # Implement softmax division by dimension/axis
        return self.softmax

    
    def backward(self, output_error, learning_rate=None):
        sm = self.softmax.reshape(-1, 1)
        jacobian = np.diagflat(sm) - np.dot(sm, sm.T)
        input_error = np.dot(jacobian, F.flatten(output_error))

        return np.reshape(input_error, self.input.shape)


if __name__ == "__main__":
    # test
    sm = Softmax()
    data = np.array([[1, 2, 3]])
    
    # forward pass
    output = sm.forward(input=data, dim=1)
    print("Softmax output:")
    print(output)

    # backward pass
    output_error = np.array([[.1, .2, .7]])
    input_gradient = sm.backward(output_error)
    print("Input gradient:")
    print(input_gradient)