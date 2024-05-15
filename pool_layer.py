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
        print(self.pool_index.shape)

        for c in range(channels):
            for row in range(out_H):
                for col in range(out_W):
                    row_index, col_index = self.pool_index[c, row * out_H + col]
                    input_error[0, c, row_index, col_index] = output_error[0, c, row, col] 

        return input_error


if __name__ == "__main__":
    # testing code
    data = np.random.randn(1, 3, 4, 4)
    maxpool = Pool(kernel_size=(2, 2)) 

    pool_result = maxpool.forward(data)
    fake_error = pool_result
    fake_gradient = maxpool.backward(fake_error)
    print(fake_gradient)