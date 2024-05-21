"""
Tanh activation layer.
"""

from activation_layer import ActivationLayer
from activations import tanh, tanh_prime


class Tanh(ActivationLayer):
    def __init__(self):
        super().__init__()

    
    def forward(self, input_data):
        self.input = input_data
        return tanh(input_data)

    
    def backward(self, output_error, learning_rate):
        return tanh_prime(output_error)