"""
NLLLoss layer.
"""

import numpy as np
from loss_layer import LossLayer


# inherit from class Layer
class NLLLoss(LossLayer):
    def __init__(self, input=None):
        self.input = input


    # returns calculate loss
    def forward(self, input, target):
        """
        Expected input shape (y_pred):
        (N, num_classes)
        Expected target shape (y_true):
        (N,) - containing class indices
        """
        self.input = input
        self.target = target
        N = input.shape[0] # batch_size

        # compute the loss
        self.loss = -np.sum()


        raise NotImplementedError

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward(self, output_error, learning_rate):
        raise NotImplementedError