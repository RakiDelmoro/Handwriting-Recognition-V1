import cupy
import torch
import math
from cupy import asnumpy
from torch.nn.functional import ctc_loss, softmax
from Model.backpropagation import backpropagation
from neural_network_layers import convolution_neurons, linear_neurons
from Model.parameters_initalization import transformer_parameters_initializer
from Model.configurations import NUM_ATTENTION_HEADS, ATTENTION_FEATURE_SIZE, NUM_LAYERS

def transformer_model(network_feature_size, conv_depth, patch_window, patches_ratio, mlp_depth, mlp_ratio, number_of_classes):
    transformer_parameters = transformer_parameters_initializer(network_feature_size, conv_depth, patch_window, patches_ratio, mlp_depth, mlp_ratio, output_class=number_of_classes)
    convolution_parameters = transformer_parameters['conv_parameters']
    mha_parameters = transformer_parameters['attn_parameters']
    encoder_mlp_parameters = transformer_parameters['enc_mlp_parameters']
    mlp_parameters = transformer_parameters['mlp_parameters']
    output_parameters = transformer_parameters['output_parameters']

    def convolution_layers(image_patches):
        activations = []
        layer_output = image_patches
        for each in range(len(convolution_parameters)):
            axons, dentrites = cupy.array(convolution_parameters[each][0]), cupy.array(convolution_parameters[each][1])
            layer_output = convolution_neurons(image_patches, axons, dentrites, step_of_patch_window=1)
            activations.append(asnumpy(layer_output))
        return activations
    
    #TODO: Maxpoold2d a function that downsample the image size and still retain important information

    def multi_head_attention(image_embeddings):
        batch_size = image_embeddings.shape[0]
        num_tokens = image_embeddings.shape[1]
        total_attn_feature_size = NUM_ATTENTION_HEADS * ATTENTION_FEATURE_SIZE
        image_projections = []
        for each in range(len(mha_parameters)):
            axons, dentrites = cupy.array(mha_parameters[each][0]), cupy.array(mha_parameters[each][1])
            projection = (cupy.matmul(image_embeddings, axons) + dentrites).reshape(batch_size, NUM_ATTENTION_HEADS, num_tokens, ATTENTION_FEATURE_SIZE)
            image_projections.append(projection)

        # attention scores -> batch | attention heads | patches | patches
        attention_scores = (cupy.matmul(image_projections[0], image_projections[1].transpose(0, 1, 3, 2))) / math.sqrt(ATTENTION_FEATURE_SIZE)
        # attention scores as probabilities
        attention_axons = cupy.array(softmax(torch.tensor(attention_scores), dim=-1))
        # image_patches_context -> batch | patches | attention heads | attention feature size
        image_patches_context = cupy.matmul(attention_axons, image_projections[2]).reshape(batch_size, num_tokens, NUM_ATTENTION_HEADS, ATTENTION_FEATURE_SIZE)
        # batch | patches | attention heads * attention feature size
        attention_output = image_patches_context.reshape(batch_size, num_tokens, total_attn_feature_size)
        return attention_output, attention_axons

    def encoder_mlp(attention_output):
        activations = []
        input_for_layer = attention_output
        for each in range(len(encoder_mlp_parameters)):
            axons, dentrites = cupy.array(encoder_mlp_parameters[each][0]), cupy.array(encoder_mlp_parameters[each][1])
            input_for_layer = cupy.matmul(input_for_layer, axons) + dentrites
            activations.append(asnumpy(input_for_layer))
        return activations

    def encoder_layer(image_embeddings):
        # Multi-Head-Attention
        attention_output, attention_axons = multi_head_attention(image_embeddings)
        # Residual connection from image embeddings to attention output
        attention_residual_connection = attention_output + image_embeddings
        # Multi-Layer-Perceptron
        mlp_activations = encoder_mlp(attention_residual_connection)
        # Residual connection from attention residual connection to mlp output
        mlp_residual_connection = cupy.array(mlp_activations[-1]) + attention_residual_connection # Second residual connection
        return mlp_residual_connection

    def encoder_forward(image_embeddings):
        encoder_activations = []
        encoder_input = image_embeddings
        for _ in range(NUM_LAYERS):
            encoder_input = encoder_layer(encoder_input)
            encoder_activations.append(asnumpy(encoder_input))
        return encoder_activations

    def multi_layer_perceptron(encoder_output):
        mlp_activations = []
        layer_input = encoder_output
        for each in range(len(mlp_parameters)):
            axons, dentrites = cupy.array(mlp_parameters[each][0]), cupy.array(mlp_parameters[each][1])
            layer_input = cupy.matmul(layer_input, axons) + dentrites
            mlp_activations.append(asnumpy(layer_input))
        return mlp_activations

    def model_output(mlp_output):
        axons, dentrites = cupy.array(output_parameters[0]), cupy.array(output_parameters[1])
        return cupy.matmul(mlp_output, axons) + dentrites

    def model_forward(image_embeddings):
        encoder_activations = encoder_forward(image_embeddings)
        mlp_activations = multi_layer_perceptron(cupy.array(encoder_activations[-1]))
        output_activation = model_output(cupy.array(mlp_activations[-1]))
        return output_activation

    return model_forward
