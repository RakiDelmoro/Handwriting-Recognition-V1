import math
import cupy
import torch
import numpy as np
from Model.configurations import IMAGE_SIZE, PATCH_SIZE,  NUM_LAYERS

def linear_initialization(input_feature, output_feature):
    weights = torch.empty((input_feature, output_feature))
    bias = torch.empty(output_feature)
    torch.nn.init.kaiming_normal_(weights, a=math.sqrt(5))
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    torch.nn.init.uniform_(bias, -bound, bound)
    return np.array(weights, dtype=cupy.float32), np.array(bias, dtype=cupy.float32)

def transformer_parameters_initializer(network_feature_size: int, mlp_architecture: list, output_class: list):
    transformer_parameters = {}

    def image_embeddings_parameters():
        image_h, image_w = IMAGE_SIZE
        patch_h, patch_w = PATCH_SIZE
        num_patches = (image_h // patch_h) * (image_w // patch_w)
        patch_feature_size = patch_h * patch_w
        patches_neurons = linear_initialization(patch_feature_size, network_feature_size)
        classification_token_parameters = np.zeros((1, 1, network_feature_size), dtype=np.float32)
        distillation_token_parameters = np.zeros((1, 1, network_feature_size), np.float32)
        position_embeddings_parameters = np.zeros((1, num_patches+2, network_feature_size), np.float32)
        parameters = [patches_neurons, classification_token_parameters, distillation_token_parameters, position_embeddings_parameters]
        transformer_parameters['image_embeddings_parameters'] = parameters

    def attention_layer_parameters():
        parameters = []
        # query, key and value projections parameters
        for _ in range(3):
            layer_weights, layer_bias = linear_initialization(network_feature_size, network_feature_size)
            parameters.append([layer_weights, layer_bias])
        transformer_parameters['attention_parameters'] = parameters

    def encoder_mlp_layer_parameters():
        parameters = []
        for layer_idx in range(len(mlp_architecture)-1):
            input_size = mlp_architecture[layer_idx]
            output_size = mlp_architecture[layer_idx+1]
            layer_parameters = linear_initialization(input_size, output_size)
            parameters.append(layer_parameters)
        transformer_parameters['encoder_mlp_parameters'] = parameters

    def mlp_layer_parameters():
        parameters = []
        for layer_idx in range(len(mlp_architecture)-1):
            input_size = mlp_architecture[layer_idx]
            output_size = mlp_architecture[layer_idx+1]
            layer_parameters = linear_initialization(input_size, output_size)
            parameters.append(layer_parameters)
        transformer_parameters['mlp_parameters'] = parameters

    def output_layer_parameters():
       parameters = linear_initialization(network_feature_size, output_class)
       transformer_parameters['output_parameters'] = parameters

    # Transformer architecture ordered parameters
    image_embeddings_parameters()
    attention_layer_parameters()
    encoder_mlp_layer_parameters()
    mlp_layer_parameters()
    output_layer_parameters()
    return transformer_parameters
