"""
NLLLoss layer.
"""

import numpy as np
from loss_layer import LossLayer


# inherit from class Layer
class NLLLoss(LossLayer):
    def __init__(self, input=None):
        self.input = input


    # returns calculate loss
    def forward(self, input, target):
        """
        !!! Expected input from LogSoftmax !!!
        Expected input shape (y_pred):
        (N, C: num_classes)
        Expected target shape (y_true):
        (N,) - containing class indices
        """
        self.input = input
        self.target = target
        N = input.shape[0] # batch_size

        # compute the loss
        self.loss = -np.sum(input[np.arange(N), target] / N)
        return self.loss


    def backward(self):
        N, C = self.input.shape
        input_error = np.zeros_like(self.input)
        input_error[np.arange(N), self.target] = -1 / N
        return input_error


if __name__ == "__main__":
    # test imports
    # utilize torch to test calculation
    import torch
    import torch.nn as nn

    # set seed
    np.random.seed(39)
    torch.manual_seed(39)

    # set default dtype
    torch.set_default_dtype(torch.float64)

    # initialize input data
    init_np = np.array([[-2.302585, -1.203972, -0.510825],
                        [-1.609438, -1.609438, -0.510825]])
    init_tensor = torch.tensor(init_np, requires_grad=True)

    # initialize target
    target_np = np.array([2, 1])
    target_tensor = torch.tensor(target_np, dtype=torch.long)

    # one forward pass
    nll = NLLLoss()
    loss_np = nll.forward(init_np, target_np)
    loss_torch = nn.functional.nll_loss(init_tensor, target_tensor)
    assert (loss_np == loss_torch)
    
    # one backward pass
    input_error = nll.backward()
    loss_torch.backward()
    assert np.alltrue(input_error == init_tensor.grad.data.numpy())
