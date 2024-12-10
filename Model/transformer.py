import cupy
import torch
import math
import numpy as np
from einops import rearrange
from cupy import asnumpy
from Model.utils import relu
from torch.nn.functional import softmax
from Model.backpropagation import backpropagation
from neural_network_layers import convolution_neurons, neurons_activations
from Model.parameters_initalization import transformer_parameters_initializer
from Model.configurations import NUM_ATTENTION_HEADS, ATTENTION_FEATURE_SIZE, NUM_LAYERS, PATCH_SIZE

def transformer_model(transformer_parameters):
    image_embeddings_parameters = transformer_parameters['image_embeddings_parameters']
    mha_parameters = transformer_parameters['attn_parameters']
    encoder_mlp_parameters = transformer_parameters['enc_mlp_parameters']
    mlp_parameters = transformer_parameters['mlp_parameters']
    output_parameters = transformer_parameters['output_parameters']

    def image_embeddings(batched_image):
        axons, dentrites = cupy.array(image_embeddings_parameters[0][0]), cupy.array(image_embeddings_parameters[0][1])
        cls_tokens = cupy.array(image_embeddings_parameters[1])
        dstl_tokens = cupy.array(image_embeddings_parameters[2])
        position_embeddings = cupy.array(image_embeddings_parameters[3])

        batched_patch_image = cupy.array(rearrange(np.array(batched_image), 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=PATCH_SIZE[0], p2=PATCH_SIZE[1]))
        patches_projection = neurons_activations(input_neurons=batched_patch_image, axons=axons, dentrites=dentrites)
        patches_activations_with_special_tokens = cupy.concatenate((cls_tokens, patches_projection, dstl_tokens), axis=1)
        patches_activations = patches_activations_with_special_tokens + position_embeddings
        return patches_activations

    def multi_head_attention(image_embeddings):
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
        attention_axons = cupy.array(softmax(torch.tensor(attention_scores), dim=-1))
        # image_patches_context -> batch | patches | attention heads | attention feature size
        image_patches_context = cupy.matmul(attention_axons, image_projections[2]).reshape(batch_size, num_tokens, NUM_ATTENTION_HEADS, ATTENTION_FEATURE_SIZE)
        # batch | patches | attention heads * attention feature size
        attention_output = image_patches_context.reshape(batch_size, num_tokens, total_attn_feature_size)
        return attention_output, image_projections, attention_axons

    def encoder_mlp(attention_output):
        activations = []
        input_for_layer = attention_output
        for each in range(len(encoder_mlp_parameters)):
            axons, dentrites = cupy.array(encoder_mlp_parameters[each][0]), cupy.array(encoder_mlp_parameters[each][1])
            input_for_layer = neurons_activations(input_for_layer, axons, dentrites, relu)
            activations.append(asnumpy(input_for_layer))
        return activations

    def encoder_layer(image_embeddings):
        # Multi-Head-Attention
        attention_output, image_projections, attention_axons = multi_head_attention(image_embeddings)
        # Residual connection from image embeddings to attention output
        attention_residual_connection = attention_output + image_embeddings
        # Multi-Layer-Perceptron
        mlp_activations = encoder_mlp(attention_residual_connection)
        # Residual connection from attention residual connection to mlp output
        mlp_residual_connection = cupy.array(mlp_activations[-1]) + attention_residual_connection # Second residual connection
        return mlp_residual_connection, image_projections, attention_axons

    def encoder_forward(image_embeddings):
        encoder_activations = []
        encoder_input = image_embeddings
        for _ in range(NUM_LAYERS):
            encoder_input, image_projections, attention_axons = encoder_layer(encoder_input)
            encoder_activations.append(asnumpy(encoder_input))
        # Append image_projections and attention axons in mha parameters for backpropagation
        mha_parameters.extend([image_projections, attention_axons])
        return encoder_activations

    def multi_layer_perceptron(encoder_output):
        mlp_activations = []
        layer_input = encoder_output
        for each in range(len(mlp_parameters)):
            axons, dentrites = cupy.array(mlp_parameters[each][0]), cupy.array(mlp_parameters[each][1])
            layer_input = neurons_activations(layer_input, axons, dentrites, relu)
            mlp_activations.append(asnumpy(layer_input))
        return mlp_activations

    def model_output(mlp_output):
        axons, dentrites = cupy.array(output_parameters[0]), cupy.array(output_parameters[1])
        return neurons_activations(mlp_output, axons, dentrites)

    def model_forward(batched_image):
        embeddings = image_embeddings(batched_image)
        encoder_activations = encoder_forward(embeddings)
        mlp_activations = multi_layer_perceptron(cupy.array(encoder_activations[-1]))
        output_activation = model_output(cupy.array(mlp_activations[-1]))
        return output_activation

    return model_forward
