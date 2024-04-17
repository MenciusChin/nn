"""
Abstract class for all types of layer;
Handles simple properties: input, output, forward and backward.
"""


class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, output_error, learning_rate):
        raise NotImplementedError
