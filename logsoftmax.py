"""
LogSoftmax activation Layer.
"""

import numpy as np
import functional as F
from activation_layer import ActivationLayer
from activations import log_softmax


class LogSoftmax(ActivationLayer):
    def __init__(self):
        super().__init__()

    
    def forward(self, input, dim):
        """
        Expected input shape:
        (N, num_classes)
        """
        self.input = input
        self.logsoftmax = log_softmax(input, dim)

        return self.logsoftmax

    
    def backward(self, output_error, learning_rate=None):
        N, C = self.input.shape
        sm = np.exp(self.logsoftmax)
        input_error = output_error - np.sum(output_error * sm, axis=1, keepdims=True)

        return input_error


if __name__ == "__main__":
    # test
    sm = LogSoftmax()
    data = np.array([[1, 2, 3],
                     [1, 2, -1]])
    
    # forward pass
    output = sm.forward(input=data, dim=1)
    print("LogSoftmax output:")
    print(output)

    # backward pass
    output_error = np.array([[.1, .2, .7],
                             [.3, .4, .3]])
    input_gradient = sm.backward(output_error)
    print("Input gradient:")
    print(input_gradient)