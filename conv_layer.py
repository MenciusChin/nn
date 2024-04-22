"""
Convolution layers
"""

import numpy as np
from layer import Layer
from multiprocessing import Pool


class Conv(Layer):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size=3,
            padding=0,
            dilation=1,
            groups=1,
            bias=None
    ) -> None:
        
        if isinstance(kernel_size, int):
            # case when kernel size is int for square kernel
            self.filters = np.random.randn(in_channels, out_channels, kernel_size, kernel_size)
        else:
            # case when kernel size is tuple
            self.filters = np.random.randn(in_channels, out_channels, kernel_size[0], kernel_size[1]) 

        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        ### Implement bias ###
    

    def pad(self, data, padding):
        """
        Helper function to add padding before convolution
        """
        data = np.insert(data, [data.shape()[1]], [0 for p in range(padding)], axis=1)
        data = np.insert(data, [0], [0 for p in range(padding)], axis=1)
        data = np.insert(data, [data.shape()[0]], [0 for p in range(padding)], axis=0)
        data = np.insert(data, [0], [0 for p in range(padding)], axis=0)
        return data


    def conv1to1(self, data, filter):
        """
        This is the function calculating 1 unit data to 1 filter,
        for later integration into convNtoM.
        Hand-select data and filter to be process.
        Take padding, dilation into consideration.
        """
        in_H, in_W = data.shape()




        
