import math
import torch
import numpy as np

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

def transformer_parameters_initializer(network_feature_size, conv_depth, patch_window_size, img_patches_magnitude, mlp_depth, mlp_ratio, output_class):
    transformer_parameters = {}

    def conv_layer_parameters():
        parameters = []
        input_channel = 1
        for each in range(1, conv_depth+1):
            output_channel = each * img_patches_magnitude
            conv_parameter = convolution_initialization((output_channel, input_channel, patch_window_size[0], patch_window_size[1]), output_channel)
            parameters.append(conv_parameter)
            input_channel = output_channel
        transformer_parameters['conv_parameters'] = parameters

    def attention_layer_parameters():
        parameters = []
        # query, key and value projections parameters
        for _ in range(3):
            projection_parameters = linear_initialization(network_feature_size, network_feature_size)
            parameters.append(projection_parameters)
        transformer_parameters['attn_parameters'] = parameters

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
        transformer_parameters['enc_mlp_parameters'] = parameters

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
    conv_layer_parameters()
    attention_layer_parameters()
    encoder_mlp_layer_parameters()
    mlp_layer_parameters()
    output_layer_parameters()
    return transformer_parameters

