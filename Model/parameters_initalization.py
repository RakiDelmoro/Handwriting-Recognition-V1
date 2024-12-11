import math
import cupy
import torch
import numpy as np
from Model.configurations import IMAGE_SIZE, PATCH_SIZE, BATCH_SIZE, NUM_LAYERS

def linear_initialization(input_feature, output_feature):
    weights = torch.empty((input_feature, output_feature))
    bias = torch.empty(output_feature)
    torch.nn.init.kaiming_normal_(weights, a=math.sqrt(5))
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    torch.nn.init.uniform_(bias, -bound, bound)
    return np.array(weights), np.array(bias)

def convolution_initialization(input_feature, output_feature):
    weights = torch.empty(input_feature)
    bias = torch.empty(output_feature)
    torch.nn.init.kaiming_normal_(weights, a=math.sqrt(5))
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    torch.nn.init.uniform_(bias, -bound, bound)
    return np.array(weights), np.array(bias)

def transformer_parameters_initializer(network_feature_size, mlp_depth, mlp_ratio, output_class):
    transformer_parameters = {}

    def image_embeddings_parameters():
        image_h, image_w = IMAGE_SIZE
        patch_h, patch_w = PATCH_SIZE
        num_patches = (image_h // patch_h) * (image_w // patch_w)
        patch_feature_size = patch_h * patch_w
        patches_neurons = linear_initialization(patch_feature_size, network_feature_size)
        classification_token_parameters = np.zeros((BATCH_SIZE, 1, network_feature_size))
        distillation_token_parameters = np.zeros((BATCH_SIZE, 1, network_feature_size))
        position_embeddings_parameters = np.zeros((BATCH_SIZE, num_patches+2, network_feature_size))
        parameters = [patches_neurons, classification_token_parameters, distillation_token_parameters, position_embeddings_parameters]
        transformer_parameters['image_embeddings_parameters'] = parameters

    def attention_layer_parameters():
        parameters = []
        # query, key and value projections parameters
        for _ in range(3):
            projection_parameters = linear_initialization(network_feature_size, network_feature_size)
            parameters.append(projection_parameters)
        return parameters

    def encoder_mlp_layer_parameters():
        parameters = []
        input_feature = network_feature_size
        for layer_idx in range(mlp_depth):
            last_layer = layer_idx == mlp_depth-1
            if last_layer: output_feature = network_feature_size
            else: output_feature = network_feature_size*mlp_ratio
            layer_parameters = linear_initialization(input_feature, output_feature)
            input_feature = output_feature
            parameters.append(layer_parameters)
        return parameters 

    def transformer_encoder_parameters():
        mha_parameters = []
        encoder_mlp_parameters = []
        for _ in range(NUM_LAYERS):
            attention_parameters = attention_layer_parameters()
            mlp_parameters = encoder_mlp_layer_parameters()
            mha_parameters.append(attention_parameters)
            encoder_mlp_parameters.append(mlp_parameters)
        transformer_parameters['encoder_parameters'] = [mha_parameters, encoder_mlp_parameters]

    def mlp_layer_parameters():
        parameters = []
        input_feature = network_feature_size
        for layer_idx in range(mlp_depth):
            last_layer = layer_idx == mlp_depth-1
            if last_layer: output_feature = network_feature_size
            else: output_feature = network_feature_size*mlp_ratio
            layer_parameters = linear_initialization(input_feature, output_feature)
            input_feature = output_feature
            parameters.append(layer_parameters)
        transformer_parameters['mlp_parameters'] = parameters

    def output_layer_parameters():
       parameters = linear_initialization(network_feature_size, output_class)
       transformer_parameters['output_parameters'] = parameters
    # Transformer architecture ordered parameters
    image_embeddings_parameters()
    transformer_encoder_parameters()
    mlp_layer_parameters()
    output_layer_parameters()
    return transformer_parameters

