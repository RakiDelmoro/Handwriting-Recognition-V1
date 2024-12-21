import cupy as c
import numpy as n
from cupy import cuda

class Neuron:
    """Neuron class for storing a scalar value and stress"""
    def __init__(self, value, axons_and_dentrites=(), operation='', device='cpu'):
        if device == 'cpu': self.value = n.array(value)
        else: self.value = c.array(value)

        self.stress = .0
        self.device = device
        self.axons_and_dentrites = set(axons_and_dentrites)
        self.backpropagation = lambda: None

    def __add__(self, other_value):
        other_value = other_value if isinstance(other_value, Neuron) else Neuron(other_value)
        out = Neuron(self.value + other_value.value, (self, other_value), '+', self.device)
        def backpropagation():
            self.stress += out.stress
            other_value.stress += out.stress
        out.backpropagation = backpropagation
        return out

    def __mul__(self, other_value):
        other_value = other_value if isinstance(other_value, Neuron) else Neuron(other_value)
        out = Neuron(self.value * other_value.value, (self, other_value), '*', self.device)
        def backpropagation():
            self.stress += out.stress
            other_value.stress += out.stress
        out.backpropagation = backpropagation
        return out

    def __repr__(self):
        return f'Neuron(value={self.value})'
    
    # TODO: Include the operation of numpy matmul
    

# def Neuron(value, axons_and_dentrites=(), operation=''):
#     '''Neuron calculations'''

#     def add(other_value):
#         pass

#     def multiply(other_value):
#         pass


input_x = Neuron(1.0)
axon = Neuron(0.5)
dentrites = Neuron(-2.0)
out = input_x + axon
print(out.value)

