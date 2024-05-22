"""
Image to column functions
"""

import numpy as np


def get_im2col_indices(
        input_shape,
        filter_shape,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1)
):
    """
    Expect input shape:
    (N, C, in_H, in_W)
    Expect filter shape:
    (f_H, f_W)
    """
    # get input shape
    N, C, in_H, in_W = input_shape
    f_H, f_W = filter_shape

    # assertion to ensure correct stride and padding
    assert (in_H + 2 * padding - dilation[0] * (f_H - 1) - 1) % stride == 0
    assert (in_W + 2 * padding - dilation[1] * (f_W - 1) - 1) % stride == 0

    # calculate output shape
    out_H = int((in_H + 2 * padding[0] - dilation[0] * (f_H - 1) - 1) / stride[0] + 1)
    out_W = int((in_W + 2 * padding[1] - dilation[1] * (f_W - 1) - 1) / stride[1] + 1)

    # get row and col indices 
    i0 = np.tile(np.repeat(np.arange(f_H), f_W), C)
    i1 = stride * np.repeat(np.arange(out_H), out_W)
    j0 = np.tile(np.arange(f_W), f_H * C)
    j1 = stride * np.tile(np.arange(out_W), out_H)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    print(i, "\n", j)


if __name__ == "__main__":
    get_im2col_indices(
        (1, 3, 5, 5),
        (3, 3)
    )