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
        """

        self.input = input

        return F.poolnto1(input, self.kernel_size, self.stride, self.padding,
                          self.dilation, self.method)

    def backward(self, output_error, learning_rate):
        return
        








if __name__ == "__main__":
    # testing code
    data = np.random.randn(1, 3, 4, 4)
    print("Original data")
    print(data)
    maxpool = Pool(kernel_size=(2, 2)) 

    pool_result = maxpool.forward(data)
    print("\nPool result")
    print(pool_result)