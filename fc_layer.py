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

        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.bias = np.random.rand(1, output_size) - 0.5 if bias else None
    
    # returns output for a given input
    def forward(self, input_data):
        if len(input_data.shape) > 1:
            # if input shape have more than 1 dim
            self.flatten = input_data
            self.input = F.flatten(input_data)
        else:
            # output = input @ weights + bias
            self.input = input_data
        self.output = self.input @ self.weights + self.bias if self.bias else self.input @ self.weights
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
        if self.bias:
            self.bias -= learning_rate * output_error

        return input_error if self.flatten is False else np.reshape(input_error, self.flatten.shape)
    

    def set_weights(self, weights):
        """
        Set weights (mostly pretrained) for current layer
        """
        self.weights = weights


if __name__ == "__main__":
    # test imports
    # utilize torch to test calculation
    import torch
    import torch.nn as nn

    # initialize input data
    # shape (1, 3, 3, 3)
    init_np = np.random.randn(1, 3, 3, 3)
    # note the dtype here needs modification
    init_tensor = torch.tensor(init_np, dtype=torch.float32)

    fc = FCLayer(27, 81, bias=False)
    flatten = nn.Flatten(0)
    linear = nn.Linear(27, 81, bias=False)
    # set same weights
    fc.set_weights(linear.weight.T.detach().numpy())
    assert np.alltrue(fc.weights == linear.weight.T.detach().numpy())
    
    # one forward pass 
    fc_output = fc.forward(init_np)
    flatten_tensor = flatten.forward(init_tensor)
    linear_output = linear.forward(flatten_tensor)
    # dtype causes numerical instability, so we round to 5th decimal
    assert np.alltrue(
        np.round(fc_output.reshape(81), 5) == np.round(linear_output.detach().numpy(), 5)
    )

    # one backward pass
    # initialze output loss
    grad_np = np.random.randn(1, 81)
    grad_tensor = torch.tensor(grad_np, dtype=torch.float32)

    input_error = fc.backward(grad_np, learning_rate=.01)
    input_grad = torch.autograd()

    