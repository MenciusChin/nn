"""
Loss layer.
"""

from layer import Layer


# inherit from base class Layer
class LossLayer(Layer):
    def __init__(self, input=None):
        self.input = input


    # returns the activated input
    def forward(self, input_data):
        raise NotImplementedError

    # returns the gradient of input
    def backward(self, output_error, learning_rate):
        raise NotImplementedError