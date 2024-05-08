"""
Convolution layers
"""

import numpy as np
from layer import Layer


class Conv(Layer):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size=3,
            stride=1,
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

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        ### Implement bias ###
    

    def pad(self, data):
        """
        Helper function to add padding before convolution
        """
        ### Look up online to optimize ###
        data = np.insert(data, [data.shape()[1]], [0 for p in range(self.padding)], axis=1)
        data = np.insert(data, [0], [0 for p in range(self.padding)], axis=1)
        data = np.insert(data, [data.shape()[0]], [0 for p in range(self.padding)], axis=0)
        data = np.insert(data, [0], [0 for p in range(self.padding)], axis=0)

        return data


    def convolve(
            self, 
            data, 
            filter,
            stride,
            dilation,
            out_r, 
            out_c
    ):
        """
        This is the function calculating 1 convolution process.
        """

        def mult(
            data, 
            filter,
            r,
            c,
            stride,
            dilation,
            out_r, 
            out_c
        ):
            """
            Helper function for 1 mult-operation in convolve.
            """
            data_r = out_r * stride + r * dilation
            data_c = out_c * stride + c * dilation
            return filter[r, c] * data[data_r, data_c]
        
        f_H, f_W = filter.shape
        output = []
        
        for r in range(f_H):
            for c in range(f_W):
                output.append(mult(data, filter, r, c, stride, dilation, out_r, out_c))

        return sum(output)
    


    def conv1to1(self, data, filter):
        """
        This is the function calculating 1 unit data to 1 filter,
        for later integration into convNtoM.
        Hand-select data and filter to be process.
        Take padding, dilation, stride... into consideration.
        """

        # pad data if needed
        if self.padding > 0:
            data = self.pad(data)

        in_H, in_W = data.shape
        f_H, f_W = filter.shape
        out_H = int((in_H + 2 * self.padding - self.dilation * (f_H - 1) - 1) / self.stride + 1)
        out_W = int((in_W + 2 * self.padding - self.dilation * (f_W - 1) - 1) / self.stride + 1)
        out_data = np.empty([out_H, out_W])

        for r in range(0, out_H):
            for c in range(0, out_W):
                out_data[r, c] = self.convolve(data=data, filter=filter, stride=self.stride, 
                                               dilation=self.dilation, out_r=r, out_c=c)
        
        return out_data


if __name__ == "__main__":

    # Testing
    data = np.ones([5, 5])
    filter = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
    
    conv = Conv(1, 1, 3)
    print(conv.conv1to1(data, filter))
    






        
