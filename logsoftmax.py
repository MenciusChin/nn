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

    
    def forward(self, input, dim=None):
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

        # modified version based on torch.nn.functional.log_softmax
        input_grad = output_error - np.sum(output_error * sm, axis=1, keepdims=True)
        input_error = sm / N
        grad_index = np.where(output_error != 0)
        input_error[grad_index] = input_grad[grad_index]

        return input_error


if __name__ == "__main__":
    # test imports
    # utilize torch to test calculation
    import torch
    import torch.nn as nn
    from nllloss import NLLLoss

    # set seed
    np.random.seed(39)
    torch.manual_seed(39)

    # set default dtype
    torch.set_default_dtype(torch.float64)

    # initialize input data
    init_np = np.random.randn(3, 3)
    init_tensor = torch.tensor(init_np, requires_grad=True)

    # one forward pass
    lsm = LogSoftmax()
    lsm_np = lsm.forward(init_np, dim=1)
    lsm_torch = nn.functional.log_softmax(init_tensor, dim=1)
    assert np.allclose(lsm_np, lsm_torch.data.numpy(), atol=1e-6)

    # initialize target
    target_np = np.array([2, 1, 0])
    target_tensor = torch.tensor(target_np, dtype=torch.long)

    # calculate loss
    nll = NLLLoss()
    loss_np = nll.forward(lsm_np, target_np)
    loss_torch = nn.functional.nll_loss(lsm_torch, target_tensor)
    assert np.allclose(loss_np, loss_torch.data.numpy(), atol=1e-6)

    # backward to logsoftmax input
    input_error = lsm.backward(nll.backward())
    loss_torch.backward()
    assert np.allclose(input_error, init_tensor.grad.data.numpy(), atol=1e-6)
