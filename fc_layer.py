"""
Fully-connected layer.
"""

import math
import numpy as np
import functional as F
from layer import Layer


# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # Expected input shape:
    # (1, input_size)
    # output_size = number of output neurons
    def __init__(self, input_size, output_size, bias=True):
        # initialize flatten to be false unless deteced
        self.flatten = None

        self.weights = np.random.randn(input_size, output_size) * math.sqrt(2 / input_size)
        self.bias = np.random.rand(1, output_size) - 0.5
    
    # returns output for a given input
    def forward(self, input_data):
        if len(input_data.shape) > 1:
            # if input shape have more than 1 dim
            self.flatten = input_data
            self.input = F.flatten(input_data)
            self.output = self.input @ self.weights + self.bias
        else:
            # output = input @ weights + bias
            self.input = input_data
            self.output = self.input @ self.weights + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward(self, output_error, learning_rate):
        """
        Expect shape of output_error:
        (N, output_shape)
        """
        # dE/dX = dE/dY * W^T
        
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error) 
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error

        return input_error if self.flatten is False else np.reshape(input_error, self.flatten.shape)


if __name__ == "__main__":
    # test
    data = np.random.randn(1, 3, 3, 3)
    fctest = FCLayer(27, 27)

    flatten_pass = fctest.forward(data)

    reshaped_gradient = fctest.backward(np.random.randn(1, 27), learning_rate=.001)
    print(reshaped_gradient)