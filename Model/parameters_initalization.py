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

def transformer_parameters_initializer(network_feature_size, mlp_depth, mlp_ratio, output_class):
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
        # Store directly on gpu or initialize parameters in CPU and move all of it into GPU
        weights = np.zeros(shape=(3, network_feature_size, network_feature_size), dtype=np.float32)
        bias = np.zeros(shape=(3, network_feature_size), dtype=np.float32)
        # query, key and value projections parameters
        for each in range(3):
            layer_weights, layer_bias = linear_initialization(network_feature_size, network_feature_size)
            weights[each, :, :] = layer_weights
            bias[each, :] = layer_bias
        transformer_parameters['attention_parameters'] = [weights, bias]

    def encoder_mlp_layer_parameters():
        weights = np.zeros(shape=(3, network_feature_size*mlp_ratio, network_feature_size*mlp_ratio))
        bias = np.zeros(shape=(3, network_feature_size*mlp_ratio))
        input_feature = network_feature_size
        for layer_idx in range(mlp_depth):
            last_layer = layer_idx == mlp_depth-1
            if last_layer: output_feature = network_feature_size
            else: output_feature = network_feature_size*mlp_ratio
            layer_weights, layer_bias = linear_initialization(input_feature, output_feature)
            weights[layer_idx, :input_feature, :output_feature] = layer_weights
            bias[layer_idx, :output_feature] = layer_bias
            input_feature = output_feature
    
        transformer_parameters['encoder_mlp_parameters'] = [weights, bias]

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
    attention_layer_parameters()
    encoder_mlp_layer_parameters()
    mlp_layer_parameters()
    output_layer_parameters()
    return transformer_parameters
