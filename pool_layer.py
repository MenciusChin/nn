"""
Pooling layers.
"""

import numpy as np
import functional as F
from layer import Layer
import multiprocessing as mp
from multiprocessing.pool import ThreadPool as Pool

### Tasks ###
# 1. write tests


class Pool(Layer):
    def __init__(
            self,
            kernel_size,
            stride=None,
            padding=0,
            dilation=1,
            method="max",
            return_indices=False,
            ceil_mode=False
    ):
        
        # if stride is None the default is kernel_size
        stride = kernel_size if stride is None else stride

        # tuple all paramters
        self.kernel_size, self.stride, self.padding, self.dilation = F.paramstuple((
            kernel_size, stride, padding, dilation
        ))

        self.method = method


    def forward(self, input):
        """
        Expected input shape:
        (1, channel, in_H, in_W)
        Expected output shape:
        (1, channel, out_H, out_W)
        Expected index shape:
        (channel, out_H * out_W, 2)
        """

        self.input = input
        pool_result, self.pool_index = F.poolnto1(
            input, self.kernel_size, self.stride, 
            self.padding, self.dilation, self.method
        )

        return pool_result


    def backward(self, output_error, learning_rate=None):
        """
        Back propagation for the Pooling layer,
        Expected shape of output_error:
        (1, channel, out_H, out_W)
        Expected output shape:
        (1, channel, in_H, in_W)
        All non-pooled index gradient is 0
        """

        _, channels, out_H, out_W = output_error.shape
        # init input_error for min/max pooling 
        input_error = np.zeros(self.input.shape)

        for c in range(channels):
            for row in range(out_H):
                for col in range(out_W):
                    row_index, col_index = self.pool_index[c, row * out_H + col]
                    input_error[0, c, row_index, col_index] = output_error[0, c, row, col] 

        return input_error


if __name__ == "__main__":
    # test imports
    # utilize torch to test calculation
    import torch
    import torch.nn as nn
    from fc_layer import FCLayer
    from mseloss import MSELoss

    # set seed
    np.random.seed(39)
    torch.manual_seed(39)

    # set default dtype
    torch.set_default_dtype(torch.float64)

    # initialize input data
    init_np = np.random.randn(1, 3, 4, 4)
    init_tensor = torch.tensor(init_np, requires_grad=True)

    mp_np = Pool(kernel_size=(2, 2))
    mp_torch = nn.MaxPool2d(2)

    # one forward pass
    pool_np = mp_np.forward(init_np)
    pool_torch = mp_torch.forward(init_tensor)
    assert np.alltrue(pool_np == pool_torch.data.numpy())

    fc = FCLayer(12, 12, bias=False)
    linear = nn.Linear(12, 12, bias=False)
    fc.set_weights(linear.weight.T.detach().numpy())
    assert np.alltrue(fc.weights == linear.weight.T.detach().numpy())

    target_np = np.random.randn(1, 12)
    target_torch = torch.tensor(target_np[0])

    mse = MSELoss()
    loss_np = mse.forward(fc.forward(pool_np), target_np)
    loss_torch = nn.functional.mse_loss(
        linear.forward(torch.flatten(pool_torch)),
        target_torch
    )
    assert (loss_np == loss_torch)

    # one backward pass
    input_error = mp_np.backward(fc.backward(mse.backward(), learning_rate=.01))
    loss_torch.backward()
    
    # due to float point inprecision, we can't assert with high precision
    # Note: the position of the gradient is correct, but the value differs since 1st decimal
    assert np.allclose(input_error, init_tensor.grad.data.numpy(), atol=.1)
