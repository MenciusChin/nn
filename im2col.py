"""
Image to column functions
Base code: https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py
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
    assert (in_H + 2 * padding[0] - dilation[0] * (f_H - 1) - 1) % stride[0] == 0
    assert (in_W + 2 * padding[1] - dilation[1] * (f_W - 1) - 1) % stride[1] == 0

    # calculate output shape
    out_H = int((in_H + 2 * padding[0] - dilation[0] * (f_H - 1) - 1) / stride[0] + 1)
    out_W = int((in_W + 2 * padding[1] - dilation[1] * (f_W - 1) - 1) / stride[1] + 1)

    # get row and col indices 
    i0 = np.repeat(np.arange(f_H), f_W)
    i0 = np.tile(i0, C)
    i1 = stride[0] * np.repeat(
        np.arange(out_H * dilation[0], step=dilation[0]), 
        out_W
    )

    j0 = np.arange(f_W)
    j0 = np.tile(j0, f_H * C)
    j1 = stride[1] * np.tile(
        np.arange(out_W * dilation[1], step=dilation[1]),
        out_H
    )

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), f_H * f_W).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(
        input,
        filter_shape,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1)
):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p0, p1 = padding
    x_padded = np.pad(input, ((0, 0), (0, 0), (p0, p1), (p0, p1)), mode='constant')

    f_H, f_W = filter_shape
    k, i, j = get_im2col_indices(
        input.shape, 
        filter_shape, 
        stride,
        padding,
        dilation
    )

    cols = x_padded[:, k, i, j]
    C = input.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(f_H * f_W * C, -1)
    return cols


"""
Not Implemented
"""
def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                                stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]
    



if __name__ == "__main__":
    img = np.random.randn(1, 3, 5, 5)
    cols = im2col_indices(img, (3, 3))
    print(cols)
