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
        



    def forward(self, input):
        return

    def backward(self, output_error, learning_rate):
        return
        








if __name__ == "__main__":
    # testing code
    print("hi")