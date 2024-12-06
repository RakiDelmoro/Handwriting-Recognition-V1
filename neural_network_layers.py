import math
import cupy
import torch
from torch.nn.functional import log_softmax
from Model.utils import softmax

def cupy_array(x):
    return cupy.round(cupy.array(x, dtype=cupy.float32), 4)

def linear_neurons(input_neurons, axons, dentrites, activation_function=None, return_parameters=False):
    if return_parameters:
        if activation_function is None:
            return cupy.matmul(input_neurons, axons, dtype=cupy.float32) + dentrites, axons, dentrites
        else:
            return activation_function(cupy.matmul(input_neurons, axons, dtype=cupy.float32) + dentrites), axons, dentrites
    else:
        if activation_function is None:
            return cupy.matmul(input_neurons, axons, dtype=cupy.float32) + dentrites
        else:
            return activation_function(cupy.matmul(input_neurons, axons, dtype=cupy.float32) + dentrites)

def convolution_neurons(input_image, axons, dentrites, step_of_patch_window):
    ''' input_neurons shape -> batch | img_channels (I will think of it as a patches) | height | width
        axons -> output_neurons_patches| input_neurons_patches | patch_window_height | patch_window_width 
        step_of_patch_window -> int '''
    batch, _, height, width = input_image.shape
    output_patches, _, patch_window_h, patch_window_w = axons.shape
    height_output = cupy.floor(1 + (height - patch_window_h) / step_of_patch_window).astype(int).item()
    width_output = cupy.floor(1 + (width - patch_window_w) / step_of_patch_window).astype(int).item()
    image_featured_extracted = cupy.zeros((batch, output_patches, height_output, width_output), dtype=cupy.float32)
    input_image_expanded = cupy.expand_dims(input_image, axis=1)
    axons_expanded = cupy.expand_dims(axons, axis=0)
    for i in range(height_output):
        for j in range(width_output):
            vertical_pixel_start = i*step_of_patch_window
            vertical_pixel_end = i*step_of_patch_window+patch_window_h
            horizontal_pixel_start = j*step_of_patch_window
            horizontal_pixel_end = j*step_of_patch_window+patch_window_w
            image_windows = input_image_expanded[:, :, :, vertical_pixel_start:vertical_pixel_end, horizontal_pixel_start:horizontal_pixel_end]
            extracted_feature_based_on_window_size = image_windows * axons_expanded
            aggregate_extracted_feature = cupy.sum(extracted_feature_based_on_window_size, axis=(2,3,4))
            image_featured_extracted[:, :, i, j] = aggregate_extracted_feature + dentrites
    return image_featured_extracted

def attention_mechanism_neurons(input_neurons, num_attn_heads, attn_head_feature_size, layer_1_connections, layer_2_connections, layer_3_connections):
    def apply_attn_heads_dim(input_data):
        # batch | patches | attn_heads | attention feature size
        new_input_data_shape = input_data.shape[:-1] + (num_attn_heads, attn_head_feature_size)
        input_data = input_data.reshape(new_input_data_shape)
        return input_data.transpose(0, 2, 1, 3)
    # batch | attn_heads | patches | attention feature size
    linear_1_projection = apply_attn_heads_dim(linear_neurons(input_neurons, axons=layer_1_connections[0], dentrites=layer_1_connections[1]))
    linear_2_projection = apply_attn_heads_dim(linear_neurons(input_neurons, axons=layer_2_connections[0], dentrites=layer_2_connections[1]))
    linear_3_projection = apply_attn_heads_dim(linear_neurons(input_neurons, axons=layer_3_connections[0], dentrites=layer_3_connections[1]))
    raw_attention_scores = (cupy.matmul(linear_1_projection, linear_2_projection.transpose(0, 1, 3, 2))) / math.sqrt(num_attn_heads*attn_head_feature_size)
    attention_axons = cupy.array(torch.nn.functional.softmax(torch.tensor(raw_attention_scores, dtype=torch.float32), dim=-1))
    # batch | attention heads | patches | attention feature size
    input_neurons_context = cupy.matmul(attention_axons, linear_3_projection)
    # batch | patches | attention heads | attention feature size
    context_input_neurons_shape = input_neurons_context.transpose(0, 2, 1, 3).shape
    reshaped_input_neurons_context = input_neurons_context.reshape(context_input_neurons_shape)
    context_input_neurons_flattened = reshaped_input_neurons_context.reshape(context_input_neurons_shape[0], context_input_neurons_shape[1], -1)
    return context_input_neurons_flattened

def multi_layer_neurons(input_neurons, depth, layers_axons, layers_dentrites):
    activation = input_neurons
    neurons_activations = [activation]
    for each in range(depth):
        axons = layers_axons[each]
        dentrites = layers_dentrites[each]
        activation = activation(cupy.matmul(activation, axons) + dentrites)
        neurons_activations.append(activation)
    return neurons_activations

def torch_convolution_neurons(input_image, weights, bias, linear_weights, stride):
    image_featured_extracted = torch.nn.functional.conv2d(input_image, weights, bias=bias, stride=stride).requires_grad_(True)
    flattened_feature_extracted = image_featured_extracted.flatten(1, -1).type(torch.float32)
    output_layer = torch.nn.functional.linear(input=flattened_feature_extracted, weight=linear_weights.transpose(0,1)).requires_grad_(True)
    return image_featured_extracted, flattened_feature_extracted, output_layer

# x_test = cupy.random.randn(1, 10, 768)
# num_attention_heads = 6
# attention_feature_size = x_test.shape[-1] // num_attention_heads
# layer_1_parameters = cupy.random.randn(768, 768), cupy.random.randn(768)
# layer_2_parameters = cupy.random.randn(768, 768), cupy.random.randn(768)
# layer_3_parameters = cupy.random.randn(768, 768), cupy.random.randn(768)

# query = x_test
# key = x_test
# value = x_test
# cupy_attention = attention_mechanism_neurons(x_test, num_attention_heads, attention_feature_size, layer_1_parameters, layer_2_parameters, layer_3_parameters)
# torch_attention, attention_weights = torch.nn.MultiheadAttention(embed_dim=768, num_heads=num_attention_heads, batch_first=True).forward(torch.tensor(query, dtype=torch.float32), torch.tensor(value, dtype=torch.float32), torch.tensor(key, dtype=torch.float32))
# print(cupy_attention)
