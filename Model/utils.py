import math
import cupy
import torch
import numpy as np
from datasets.utils import char_to_index, PAD_TOKEN

def unpadded_length_tokens(word_tokens):
    length = (word_tokens != char_to_index[PAD_TOKEN]).sum(1)
    return length

def leaky_relu(input_data, return_derivative=False):
    if return_derivative: return cupy.where(input_data > 0, 1, 0.05 * input_data)
    else: return cupy.maximum(input_data * 0.05, input_data)

def relu(input_data, return_derivative=False):
    if return_derivative: return cupy.where(input_data > 0, 1, 0)
    else: return cupy.maximum(0, input_data)

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

def softmax(input_data, layer_stress=None, return_derivative=False):
    # Subtract max value for numerical stability
    shifted_data = input_data - cupy.max(input_data, axis=-1, keepdims=True)
    # Calculate exp
    exp_data = cupy.exp(shifted_data)
    # Sum along axis=1 and keep dimensions for broadcasting
    sum_exp_data = cupy.sum(exp_data, axis=-1, keepdims=True)
    if return_derivative:
        dot_product = np.dot(input_data, layer_stress)
        return input_data * (layer_stress - dot_product)
    else:
        return exp_data / sum_exp_data

def log_softmax(input_data):
    # Subtract max value for numerical stability
    shifted_data = input_data - cupy.max(input_data, axis=-1, keepdims=True)
    # Calculate exp
    exp_data = cupy.exp(shifted_data)
    # Sum along axis=1 and keep dimensions for broadcasting
    log_sum_exp = cupy.log(cupy.sum(exp_data, axis=-1, keepdims=True))
    return shifted_data - log_sum_exp

def cross_entropy_loss(model_prediction, expected_indices):
    batch_size = model_prediction.shape[0]
    prediction_as_probabilities = softmax(model_prediction)
    model_prediction = prediction_as_probabilities[:, 0, :]
    correct_class_mask = cupy.zeros_like(model_prediction)
    correct_class_mask[cupy.arange(batch_size), expected_indices] = 1.0
    loss = cupy.mean(-cupy.sum(correct_class_mask * cupy.log(model_prediction), axis=1))
    
    layer_stress = prediction_as_probabilities.copy()
    layer_stress[cupy.arange(batch_size), 0, expected_indices] -= 1.0
    return loss, layer_stress

def generate_square_mask(size):
    mask = (cupy.triu(cupy.ones((size, size))) == 1)
    mask = mask.astype(float)
    mask = cupy.where(mask == 0, float('-inf'), 0.).transpose()
