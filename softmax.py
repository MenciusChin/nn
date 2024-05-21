"""
Softmax activation Layer.
"""

import numpy as np
import functional as F
from activation_layer import ActivationLayer
from activations import softmax


class Softmax(ActivationLayer):
    def __init__(self):
        super().__init__()

    
    def forward(self, input, dim):
        """
        Expected input shape:
        (N, num_classes)
        """
        self.input = input
        self.softmax = softmax(input, dim)

        return self.softmax

    
    def backward(self, output_error, learning_rate=None):
        # reshape softmax into column vector (N * num_classes, 1)
        sm = self.softmax.reshape(-1, 1)
        # The jacobian matrix with expected shape:
        # (N * num_classes, N * num_classes)
        jacobian = np.diagflat(sm) - np.dot(sm, sm.T)
        # expect shape of input_error:
        # (N * num_classes, 1)
        input_error = np.dot(jacobian, F.flatten(output_error))

        return np.reshape(input_error, self.input.shape)


if __name__ == "__main__":
    # test imports
    # utilize torch to test calculation
    import torch
    import torch.nn as nn
    from nllloss import NLLLoss

    # set seed
    np.random.seed(39)
    torch.manual_seed(39)

    # set default dtype
    torch.set_default_dtype(torch.float64)

    # initialize input data
    init_np = np.random.randn(2, 20)
    init_tensor = torch.tensor(init_np, requires_grad=True)

    # one forward pass
    sm = Softmax()
    sm_np = sm.forward(init_np, dim=1)
    sm_torch = nn.functional.softmax(init_tensor, dim=1)
    assert np.allclose(sm_np, sm_torch.data.numpy(), atol=1e-6)

    # calculate loss 
    # one backward pass
    