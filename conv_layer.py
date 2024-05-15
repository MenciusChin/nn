"""
Convolution layers
"""

import numpy as np
from layer import Layer
import functional as F
import multiprocessing as mp
# This is for windows
from multiprocessing.pool import ThreadPool as Pool

### Tasks ###
# 1. run full test with mnist
# 2. stride dilation calculation when doing full-convolution
# 3. grouping 
# 4. check multiprocessing


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
        
        # reformat paramters into tuples
        kernel_size, self.stride, self.padding, self.dilation = F.paramtuple([
            kernel_size, stride, padding, dilation
        ])
        
        # initalize filters
        self.filters = np.random.randn(in_channels, out_channels, kernel_size[0], kernel_size[1])

        ### add grouping ###
        self.groups = groups
        
        # introduce bias for smaller network
        self.bias = True if bias else False


    def forward(self, input):
        """
        Forward pass of the Convolution layer,
        expected input shape:
        (1, in_channel, in_H, in_W)
        """

        self.input = input
        self.output = F.convntom(
            input,
            self.filters,
            self.stride,
            self.padding,
            self.dilation
        )

        # initialize bias at the very first forward pass 
        if self.bias is True:
            self.bias = np.random.randn(1, self.output.shape[1], self.output.shape[2], self.output.shape[3])
        elif self.bias is False:
            return self.output
        
        return (self.output + self.bias)
    

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
                filter_input.append((
                    self.input[0, ic], 
                    output_error[0, oc],
                    self.stride,
                    self.padding,
                    self.dilation
                ))

        with Pool(mp.cpu_count()) as p:
            filter_error = p.starmap(F.conv1to1, filter_input)
            p.terminate()
            p.join()
        
        filter_error = np.reshape(
            np.array(filter_error),
            [in_channel, out_channel, self.filters.shape[2], self.filters.shape[3]]
        )

        # update filter gradient
        self.filters -= learning_rate * filter_error

        # get input gradient
        input_error = F.convntom(
            output_error, 
            self.filters,
            ### check stride calculation ###
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            full=True,
            input_shape=(self.input.shape[2], self.input.shape[3])
        )

        # update bias
        if self.bias is not False:
            self.bias -= learning_rate * output_error

        return input_error



if __name__ == "__main__":
    # Testing
    data = np.random.randn(1, 3, 7, 7)
    conv = Conv(3, 4, 3)

    fake_error = conv.forward(data)
    print(fake_error.shape)
    fake_gradient = conv.backward(fake_error, learning_rate=.01)
    print(fake_gradient.shape)
    






        
