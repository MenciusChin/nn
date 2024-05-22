"""
Convolution layers
"""

import numpy as np
from layer import Layer
import functional as F
import multiprocessing as mp
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
        kernel_size, self.stride, self.padding, self.dilation = F.paramstuple([
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
    

    def set_filters(self, filters):
        """
        Set filters (mostly pretrained) for current layer
        """
        self.filters = filters



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
    init_np = np.random.randn(1, 3, 5, 5)
    init_tensor = torch.tensor(init_np, requires_grad=True)

    conv_np = Conv(3, 5, 3)
    conv_torch = nn.Conv2d(3, 5, 3, bias=False)
    conv_np.set_filters(np.swapaxes(conv_torch.weight.detach().numpy(), 0, 1))
    assert np.alltrue(conv_np.filters == np.swapaxes(conv_torch.weight.detach().numpy(), 0, 1))


    # one forward pass
    conved_np = conv_np.forward(init_np)
    conved_torch = conv_torch.forward(init_tensor)
    assert np.allclose(conved_np, conved_torch.data.numpy(), atol=1e-6)

    fc = FCLayer(45, 45, bias=False)
    linear = nn.Linear(45, 45, bias=False)
    fc.set_weights(linear.weight.T.detach().numpy())
    assert np.alltrue(fc.weights == linear.weight.T.detach().numpy())

    target_np = np.random.randn(1, 45)
    target_torch = torch.tensor(target_np[0])

    mse = MSELoss()
    loss_np = mse.forward(fc.forward(conved_np), target_np)
    loss_torch = nn.functional.mse_loss(
        linear.forward(torch.flatten(conved_torch)),
        target_torch
    )
    assert np.allclose(loss_np, loss_torch.data.numpy(), atol=1e-6)

    # one backward pass
    input_error = conv_np.backward(fc.backward(mse.backward(), learning_rate=.01), learning_rate=.01)
    loss_torch.backward()
    
    # due to float point inprecision, we can't assert with high precision
    print(input_error)
    print("")
    print(init_tensor.grad.data.numpy())
    assert np.allclose(input_error, init_tensor.grad.data.numpy(), atol=1e-2)
