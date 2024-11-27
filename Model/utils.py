import math
import cupy
import torch
from datasets.utils import characters_to_ints, PAD_TOKEN

def unpadded_length_tokens(word_tokens):
    length = (word_tokens != characters_to_ints(PAD_TOKEN)).sum(1)
    return length

def leaky_relu(input_data, return_derivative=False):
    if return_derivative:
        return cupy.where(input_data > 0, 1, 0.05 * input_data)
    else:
        return cupy.maximum(input_data * 0.05, input_data)

def relu(input_data, return_derivative=False):
    if return_derivative:
        return cupy.where(input_data > 0, 1, 0)
    else:
        return cupy.maximum(0, input_data)

def sigmoid(input_data, return_derivative=False):
    if return_derivative:
       input_data = 1.0 / (1.0+cupy.exp(-input_data))
       return input_data * (1 - input_data)
    else:
        return 1.0 / (1.0+cupy.exp(-input_data))

def tanh(input_data, return_derivative=False):
    if return_derivative:
        input_data = (cupy.exp(input_data) - cupy.exp(-input_data))/(cupy.exp(input_data) + cupy.exp(-input_data))
        return 1 - input_data * input_data
    else:
        return (cupy.exp(input_data) - cupy.exp(-input_data))/(cupy.exp(input_data) + cupy.exp(-input_data))

def axons_and_dentrites_initialization(input_feature, output_feature):
    weights = torch.empty((input_feature, output_feature))
    bias = torch.empty(output_feature)
    torch.nn.init.kaiming_normal_(weights, a=math.sqrt(5))
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    torch.nn.init.uniform_(bias, -bound, bound)
    return cupy.array(weights), cupy.array(bias)

def softmax(input_data):
    # Subtract max value for numerical stability
    shifted_data = input_data - cupy.max(input_data, axis=-1, keepdims=True)
    # Calculate exp
    exp_data = cupy.exp(shifted_data)
    # Sum along axis=1 and keep dimensions for broadcasting
    sum_exp_data = cupy.sum(exp_data, axis=-1, keepdims=True)
    return exp_data / sum_exp_data

def generate_square_mask(size):
    mask = (cupy.triu(cupy.ones((size, size))) == 1)
    mask = mask.astype(float)
    mask = cupy.where(mask == 0, float('-inf'), 0.).transpose()
