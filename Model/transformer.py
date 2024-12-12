import cupy
import torch
import math
import numpy as np
from einops import rearrange
from cupy import asnumpy
from Model.utils import relu
from Model.utils import softmax
from Model.backpropagation import backpropagation
from neural_network_layers import convolution_neurons, neurons_activations
from Model.parameters_initalization import transformer_parameters_initializer
from Model.configurations import NUM_ATTENTION_HEADS, ATTENTION_FEATURE_SIZE, NUM_LAYERS, PATCH_SIZE

def transformer_model(transformer_parameters):
    image_embeddings_parameters = transformer_parameters['image_embeddings_parameters']
    encoder_parameters = transformer_parameters['encoder_parameters']
    mlp_parameters = transformer_parameters['mlp_parameters']
    output_parameters = transformer_parameters['output_parameters']

    transformer_model_activations = {}

    def image_embeddings(batched_image):
        axons, dentrites = image_embeddings_parameters[0][0], image_embeddings_parameters[0][1]
        cls_tokens = image_embeddings_parameters[1]
        dstl_tokens = image_embeddings_parameters[2]
        position_embeddings = image_embeddings_parameters[3]

        batched_patch_image = cupy.array(rearrange(batched_image), 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=PATCH_SIZE[0], p2=PATCH_SIZE[1])
        patches_projection = cupy.matmul(batched_patch_image, axons) + dentrites
        patches_activations_with_special_tokens = cupy.concatenate((cls_tokens, patches_projection, dstl_tokens), axis=1)
        patches_activations = patches_activations_with_special_tokens + position_embeddings
        transformer_model_activations['input_previous_activations'] = batched_patch_image
        return patches_activations

    def multi_head_attention(image_embeddings, mha_parameters):
        batch_size = image_embeddings.shape[0]
        num_tokens = image_embeddings.shape[1]
        total_attn_feature_size = NUM_ATTENTION_HEADS * ATTENTION_FEATURE_SIZE
        image_projections = []
        for each in range(len(mha_parameters)):
            axons, dentrites = cupy.array(mha_parameters[each][0]), cupy.array(mha_parameters[each][1])
            projection = neurons_activations(image_embeddings, axons, dentrites).reshape(batch_size, NUM_ATTENTION_HEADS, num_tokens, ATTENTION_FEATURE_SIZE)
            image_projections.append(projection)
        # attention scores -> batch | attention heads | patches | patches
        attention_scores = (cupy.matmul(image_projections[0], image_projections[1].transpose(0, 1, 3, 2))) / math.sqrt(ATTENTION_FEATURE_SIZE)
        # attention scores as probabilities
        attention_axons = softmax(attention_scores)
        # image_patches_context -> batch | patches | attention heads | attention feature size
        image_patches_context = cupy.matmul(attention_axons, image_projections[2]).reshape(batch_size, num_tokens, NUM_ATTENTION_HEADS, ATTENTION_FEATURE_SIZE)
        # batch | patches | attention heads * attention feature size
        attention_output = image_patches_context.reshape(batch_size, num_tokens, total_attn_feature_size)
        return attention_output, image_projections, attention_axons

    def encoder_mlp(attention_output, encoder_mlp_parameters):
        encoder_mlp_activations = [attention_output]
        input_for_layer = attention_output
        for each in range(len(encoder_mlp_parameters)):
            axons, dentrites = cupy.array(encoder_mlp_parameters[each][0]), cupy.array(encoder_mlp_parameters[each][1])
            input_for_layer = cupy.matmul(input_for_layer, axons) + dentrites
            if each == len(encoder_mlp_parameters)-1: continue
            encoder_mlp_activations.append(asnumpy(input_for_layer))
        transformer_model_activations['encoder_mlp_previous_activations'] = encoder_mlp_activations
        return input_for_layer

    def encoder_layer(image_embeddings, attention_parameters, encoder_mlp_parameters):
        # Multi-Head-Attention
        attention_output, image_projections, attention_axons = multi_head_attention(image_embeddings, attention_parameters)
        # Residual connection from image embeddings to attention output
        attention_residual_connection = attention_output + image_embeddings
        # Multi-Layer-Perceptron
        mlp_output = encoder_mlp(attention_residual_connection, encoder_mlp_parameters)
        # Residual connection from attention residual connection to mlp output
        mlp_residual_connection = mlp_output + attention_residual_connection
        return mlp_residual_connection, image_projections, attention_axons

    def encoder_forward(image_embeddings):
        attentions_axons = []
        mha_activations = [asnumpy(image_embeddings)]
        encoder_input = image_embeddings
        for each in range(NUM_LAYERS):
            attention_parameters = encoder_parameters[0][each]
            encoder_mlp_parameters = encoder_parameters[1][each]
            encoder_input, image_projections, attention_axons = encoder_layer(encoder_input, attention_parameters, encoder_mlp_parameters)
            attentions_axons.append(attention_axons)
            if each == NUM_LAYERS-1: continue
            mha_activations.append(asnumpy(encoder_input))
        transformer_model_activations['mha_previous_activations'] = mha_activations
        return encoder_input, image_projections, attentions_axons

    def multi_layer_perceptron(encoder_output):
        mlp_activations = [encoder_output]
        layer_input = encoder_output
        for each in range(len(mlp_parameters)):
            axons, dentrites = cupy.array(mlp_parameters[each][0]), cupy.array(mlp_parameters[each][1])
            layer_input = cupy.matmul(layer_input, axons) + dentrites
            if each == len(mlp_parameters)-1: continue
            mlp_activations.append(asnumpy(layer_input))
        transformer_model_activations['mlp_previous_activations'] = mlp_activations
        return layer_input

    def model_output(mlp_output):
        axons, dentrites = cupy.array(output_parameters[0]), cupy.array(output_parameters[1])
        output_prediction = neurons_activations(mlp_output, axons, dentrites)
        transformer_model_activations['model_output_previous_activation'] = mlp_output
        return output_prediction

    def model_forward(batched_image):
        embeddings = image_embeddings(batched_image)
        encoder_output, image_projections, attention_axons = encoder_forward(embeddings)
        mlp_output = multi_layer_perceptron(encoder_output)
        output_activation = model_output(mlp_output)
        return output_activation, transformer_model_activations, image_projections, attention_axons

    return model_forward
