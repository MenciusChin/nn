"""
Helper functions, for now
"""

import numpy as np
from layer import Layer
import multiprocessing as mp
# This is for windows
from multiprocessing.pool import ThreadPool as Pool


def intuple(param):
    """
    This reformat the input parameter,
    returning (int, int)
    """

    return (param, param) if isinstance(param, int) else param


def paramtuple(params):
    """
    This reformat input all input parameters
    into tuple (int, int)
    """

    with Pool(mp.cpu_count()) as p:
        output_params = p.map(intuple, params)
        p.terminate()
        p.join()
    
    return output_params


def pad(data, padding):
    """
    Helper function to add padding,
    input is expected as tuple
    """

    return np.pad(data, ((padding[0]), (padding[1])), mode="constant")


def mult(
        data, 
        filter,
        stride,
        dilation,
        r,
        c,
        out_r, 
        out_c
):
    """
    Helper function for 1 mult-operation in convolve.
    """

    data_r = out_r * stride[0] + r * dilation[0]
    data_c = out_c * stride[1] + c * dilation[1]

    return filter[r, c] * data[data_r, data_c]


def convolve(
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
    
    f_H, f_W = filter.shape
    input = []
    
    for r in range(f_H):
        for c in range(f_W):
            input.append((
                data,
                filter,
                stride,
                dilation,
                r,
                c,
                out_r,
                out_c
            ))
    
    with Pool(mp.cpu_count()) as p:
        output = p.starmap(mult, input)
        p.terminate()
        p.join()

    return sum(output)


def conv1to1(
        data, 
        filter,
        stride,
        padding,
        dilation,
        full=False,
        input_shape=None
):
    """
    This is the function calculating 1 unit data to 1 filter,
    for later integration into convNtoM.
    Hand-select data and filter to be process.
    Take padding, dilation, stride... into consideration.
    """
    
    # if full-convolution, flip the filter then switch data and filter
    # since the calculation is:
    # full-convolution(filter, output_error)
    # also calculate padding
    if full:
        filter = np.flip(filter, axis=0)
        filter = np.flip(filter, axis=1)

        padding = (filter.shape[0] - 1, filter.shape[1] - 1)

    # pad data if needed
    data = pad(data, padding)

    in_H, in_W = data.shape
    f_H, f_W = filter.shape

    # if full-convolution, out_H and out_W would match the input_shape
    if full:
        out_H, out_W = input_shape[0], input_shape[1]
    else:
        out_H = int((in_H + 2 * padding[0] - dilation[0] * (f_H - 1) - 1) / stride[0] + 1)
        out_W = int((in_W + 2 * padding[1] - dilation[1] * (f_W - 1) - 1) / stride[1] + 1)
    input = []

    for r in range(0, out_H):
        for c in range(0, out_W):
            input.append((data, filter, stride, dilation, r, c))
    
    with Pool(mp.cpu_count()) as p:
        out_data = p.starmap(convolve, input)
        p.terminate()
        p.join()
    
    out_data = np.array(out_data)

    return np.reshape(out_data, [out_H, out_W])


def convntom(
        data, 
        filters,
        stride,
        padding,
        dilation,
        full=False,
        input_shape=None
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
        filters = np.swapaxes(filters, 0, 1)

    input = []
    in_channels, out_channels = filters.shape[0], filters.shape[1]

    for oc in range(out_channels):
        for ic in range(in_channels):
            input.append((
                data[0, ic], 
                filters[ic, oc],
                stride,
                padding,
                dilation,
                full,
                input_shape
            ))
    
    with Pool(mp.cpu_count()) as p:
        out_data = p.starmap(conv1to1, input)
        p.terminate()
        p.join()
    
    out_H, out_W = out_data[0].shape[0], out_data[0].shape[1]
    # sum up result from in_channels to one combined result
    out_data = np.sum(np.reshape(np.array(out_data), 
                                    [out_channels, in_channels, out_H, out_W]), axis=1)
    
    return np.expand_dims(out_data, axis=0)