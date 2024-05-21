"""
MSELoss layer.
"""

import numpy as np
from loss import mse, mse_prime
from loss_layer import LossLayer


# inherit from class Layer
class MSELoss(LossLayer):
    def __init__(self, input=None):
        self.input = input


    # returns calculate loss
    def forward(self, input, target):
        """
        Expected input shape (y_pred):
        (N, num_classes)
        Expected target shape (y_true):
        (N,) - containing class indices
        """
        self.input = input
        self.target = target
        return mse(self.target, self.input)


    def backward(self):
        return mse_prime(self.target, self.input)
    

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

    # initialize y_pred
    # shape (batch_size, output_size)
    pred_np = np.random.randn(1, 20)
    pred_tensor = torch.tensor(pred_np, requires_grad=True)

    # initialize y_true
    true_np = np.array(np.random.randint(0, 2, size=(1, 20)), dtype="float64")
    true_tensor = torch.tensor(true_np)

    # one mse forward
    msel = MSELoss()
    loss_np = msel.forward(pred_np, true_np)
    loss_torch = nn.functional.mse_loss(pred_tensor, true_tensor)
    # here we use allclose for numerical stability
    assert np.allclose(loss_np, loss_torch.data.numpy(), atol=1e-6)

    # one mse backward
    input_error = msel.backward()
    loss_torch.backward()
    assert np.allclose(input_error, pred_tensor.grad.data.numpy(), atol=1e-6)
    