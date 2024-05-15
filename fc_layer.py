"""
Fully-connected layer.
"""

import math
import numpy as np
import functional as F
from layer import Layer

### Tasks ###
# 1. add auto flatten when input layer dim > 1
# 2. add backwards for flatten



# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size, bias=True):
        self.weights = np.random.randn(input_size, output_size) * math.sqrt(2 / input_size)
        self.bias = np.random.rand(1, output_size) - 0.5
    
    # returns output for a given input
    def forward(self, input_data):
        self.input = input_data
        # output = input @ weights + bias
        self.output = self.input @ self.weights + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward(self, output_error, learning_rate):
        # dE/dX = dE/dY * W^T
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error