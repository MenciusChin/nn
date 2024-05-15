"""
Pooling layers.
"""

import numpy as np
import functional as F
from layer import Layer
import multiprocessing as mp
# This is for windows
from multiprocessing.pool import ThreadPool as Pool


class MaxPool(Layer):
    def __init__(
            self,
            kernel_size,
            stride=None,
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False
    ):
        
        # if stride is None the default is kernel_size
        stride = kernel_size if stride is None else stride

        # tuple all paramters
        self.kernel_size, self.stride, self.padding, self.dilation = F.paramstuple(
            kernel_size, stride, padding, dilation
        )

    def forward(self, input):
        """
        Expected input shape:
        (1, channel, in_H, in_W)
        Expected output shape:
        (1, channel, out_H, out_W)
        """

        

        return

    def backward(self, output_error, learning_rate):
        return
        








if __name__ == "__main__":
    # testing code
    print("hi")