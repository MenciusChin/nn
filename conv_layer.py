"""
Convolution layers
"""

import numpy as np
from layer import Layer
import multiprocessing as mp
# This is for windows
from multiprocessing.pool import ThreadPool as Pool


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
            bias=False
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


    def forward(self, input):
        """
        Forward pass of the Convolution layer,
        expected input shape:
        (1, in_channel, in_H, in_W)
        """
        self.input = input
        self.output = self.convntom(
            input,
            self.filters,
            self.padding,
            self.dilation,
            self.stride
        )
        return self.output
    

    def backward(self, output_error, learning_rate):
        """
        Back propagation for the Convolution layer,
        for every element of filter F, output O:
        dL/dFi = sum(dL/dOk * dOk/dFi) -> dL/dF = X (input) @ dL/dO (output_error)
        The expected shape of output_error would be:
        (1, out_channel, out_H, out_W) same as the shape if out_data
        """
        
        # get filter gradient
        filter_input = []
        in_channel, out_channel = self.filters.shape[0], self.filters.shape[1]

        for ic in range(in_channel):
            for oc in range(out_channel):
                filter_input.append((input[0, ic], output_error[0, oc]))

        with Pool(mp.cpu_count()) as p:
            filter_error = p.starmap(self.conv1to1, filter_input)
            p.terminate()
            p.join()
        
        filter_error = np.reshape(
            np.array(filter_error),
            [in_channel, out_channel, self.filters.shape[2], self.filters.shape[3]]
        )

        # update filter gradient
        self.filters -= learning_rate * filter_error

        # get input gradient


        

            
        






    def pad(self, data, padding):
        """
        Helper function to add padding before convolution
        """
        if isinstance(padding, int):
            # case when padding is int for square kernel
            row_pad = col_pad = padding
        else:
            # case when padding is tuple
            row_pad, col_pad = padding
        
        ### Look up online to optimize ###
        data = np.insert(data, [data.shape()[1]], [0 for p in range(row_pad)], axis=1)
        data = np.insert(data, [0], [0 for p in range(row_pad)], axis=1)
        data = np.insert(data, [data.shape()[0]], [0 for p in range(col_pad)], axis=0)
        data = np.insert(data, [0], [0 for p in range(col_pad)], axis=0)

        return data


    def mult(
            self,
            data, 
            filter,
            r,
            c,
            out_r, 
            out_c
    ):
        """
        Helper function for 1 mult-operation in convolve.
        """

        data_r = out_r * self.stride + r * self.dilation
        data_c = out_c * self.stride + c * self.dilation
        return filter[r, c] * data[data_r, data_c]


    def convolve(
            self, 
            data, 
            filter,
            out_r, 
            out_c
    ):
        """
        This is the function calculating 1 convolution process.
        """
        
        f_H, f_W = filter.shape
        input = []
        
        for r in range(f_H):
            for c in range(f_W):
                input.append((data, filter, r, c, out_r, out_c))
        
        with Pool(mp.cpu_count()) as p:
            output = p.starmap(self.mult, input)
            p.terminate()
            p.join()

        return sum(output)
    

    def conv1to1(
            self, 
            data, 
            filter,
            padding,
            dilation,
            stride,
            full=False
    ):
        """
        This is the function calculating 1 unit data to 1 filter,
        for later integration into convNtoM.
        Hand-select data and filter to be process.
        Take padding, dilation, stride... into consideration.
        """
        
        # if full-convolution, flip the current filter
        if full:
            filter = np.flip(filter, axis=0)
            filter = np.flip(filter, axis=1)

        # pad data if needed
        if padding > 0:
            data = self.pad(data, padding)

        in_H, in_W = data.shape
        f_H, f_W = filter.shape
        out_H = int((in_H + 2 * padding - dilation * (f_H - 1) - 1) / stride + 1)
        out_W = int((in_W + 2 * padding - dilation * (f_W - 1) - 1) / stride + 1)
        input = []

        for r in range(0, out_H):
            for c in range(0, out_W):
                input.append((data, filter, r, c))
        
        with Pool(mp.cpu_count()) as p:
            out_data = p.starmap(self.convolve, input)
            p.terminate()
            p.join()
        
        out_data = np.array(out_data)

        return np.reshape(out_data, [out_H, out_W])
    

    def convntom(
            self, 
            data, 
            filters,
            padding,
            dilation,
            stride,
            full=False
    ):
        """
        This is the function calculating convolution for 
        n input channels to m output channels.
        The input data shape is expected as: 
        (1, in_channel, in_H, in_W)
        The filters shape is expected as:
        (in_channel, out_channel, f_H, f_W)
        The output shape is expected as:
        (1, out_channel, out_H, out_W)
        """

        # if full-convolution, flip in/out_channel
        if full:
            filters = np.array()

        input = []
        in_channels, out_channels = filters.shape[0], filters.shape[1]

        for oc in range(out_channels):
            for ic in range(in_channels):
                input.append((
                    data[0, ic], 
                    filters[ic, oc],
                    padding,
                    dilation,
                    stride
                ))
        
        with Pool(mp.cpu_count()) as p:
            out_data = p.starmap(self.conv1to1, input)
            p.terminate()
            p.join()
        
        out_H, out_W = out_data[0].shape[0], out_data[0].shape[1]
        # sum up result from in_channels to one combined result
        out_data = np.sum(np.reshape(np.array(out_data), 
                                     [out_channels, in_channels, out_H, out_W]), axis=1)
        
        return np.expand_dims(out_data, axis=0)


if __name__ == "__main__":
    # Testing
    data = np.random.randn(1, 3, 7, 7)
    conv = Conv(3, 4, 3)

    (res := conv.forward(data))
    print(res.shape)
    print(res)
    






        
