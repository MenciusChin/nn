"""
Activation layer.
"""

from layer import Layer


# inherit from base class Layer
class ActivationLayer(Layer):
    def __init__(self, input=None):
        self.input = input


    # returns the activated input
    def forward(self, input_data):
        raise NotImplementedError

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward(self, output_error, learning_rate):
        raise NotImplementedError