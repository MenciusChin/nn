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
        self.output = self.input @ self.weights if self.bias is False else self.input @ self.weights + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward(self, output_error, learning_rate, train=True):
        """
        Expect shape of output_error:
        (N, output_shape)
        """
        # dE/dX = dE/dY * W^T
        
        input_error = output_error @ self.weights.T
        weights_error = self.input.T @ output_error
        # dBias = output_error

        # update parameters if training
        if train:
            self.weights -= learning_rate * weights_error
            if self.bias:
                self.bias -= learning_rate * output_error

        return input_error if self.flatten is False else np.reshape(input_error, self.flatten.shape)
    

    def set_weights(self, weights):
        """
        Set weights (mostly pretrained) for current layer
        """
        self.weights = weights

    
    def set_bias(self, bias):
        """
        Set bias (mostly pretrained) for current layer
        """
        self.bias = bias


if __name__ == "__main__":
    # test imports
    # utilize torch to test calculation
    import torch
    import torch.nn as nn
    from loss import mse, mse_prime

    # set seed
    np.random.seed(39)
    torch.manual_seed(39)

    # set default dtype
    torch.set_default_dtype(torch.float64)

    # initialize input data
    # shape (1, 3, 3, 3)
    init_np = np.random.randn(1, 3, 3, 3)
    init_tensor = torch.tensor(init_np, requires_grad=True)

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
    assert np.alltrue(fc_output.reshape(81) == linear_output.detach().numpy())

    # define a simple MSE loss
    # set y_true
    target_np = np.random.randn(1, 81)
    target_torch = torch.tensor(target_np[0])

    # output loss
    loss_np = mse(target_np, fc_output)
    loss_torch = nn.functional.mse_loss(linear_output, target_torch)
    assert (loss_np == loss_torch.data.numpy())

    # one backward pass
    output_error = mse_prime(target_np, fc_output)
    input_error = fc.backward(output_error, learning_rate=.01)

    loss_torch.backward()

    # Extract gradients
    # due to float-point inaccuracies, use tolerance instead
    assert np.allclose(input_error, init_tensor.grad.data.numpy(), atol=1e-2)