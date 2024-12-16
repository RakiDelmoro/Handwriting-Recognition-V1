import cupy
import torch
import math
import numpy as np
from einops import rearrange
from cupy import asnumpy
from Model.utils import softmax, relu
from Model.configurations import NUM_ATTENTION_HEADS, ATTENTION_FEATURE_SIZE, NUM_LAYERS, PATCH_SIZE, BATCH_SIZE, NUM_PATCHES, NETWORK_FEATURE_SIZE, MLP_ARCHITECTURE

def transformer_model(transformer_parameters):
    transformer_model_activations = {}

    def image_embeddings(batched_image):
        image_embeddings_parameters = transformer_parameters['image_embeddings_parameters']
        learnable_tokens = transformer_parameters['learnable_tokens']

        axons, dentrites = cupy.array(image_embeddings_parameters[0]), cupy.array(image_embeddings_parameters[1])
        cls_tokens = cupy.resize(learnable_tokens[0][:, 0, :], new_shape=(BATCH_SIZE, 1, NETWORK_FEATURE_SIZE))
        dstl_tokens = cupy.resize(learnable_tokens[1][:, 0, :], new_shape=(BATCH_SIZE, 1, NETWORK_FEATURE_SIZE))
        # Position embeddings have a +2 for num patches since we have a 2 special tokens (classification token & distillation token)
        position_embeddings = cupy.resize(learnable_tokens[2], new_shape=(BATCH_SIZE, NUM_PATCHES+2, NETWORK_FEATURE_SIZE))
        batched_patch_image = cupy.array(rearrange(batched_image, 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=PATCH_SIZE[0], p2=PATCH_SIZE[1]))
        patches_projection = cupy.matmul(batched_patch_image, axons) + dentrites[0, :]
        patches_activations_with_special_tokens = cupy.concatenate((cls_tokens, patches_projection, dstl_tokens), axis=1)
        patches_activations = patches_activations_with_special_tokens + position_embeddings
        transformer_model_activations['input_previous_activations'] = asnumpy(batched_patch_image)
        return patches_activations

    def multi_head_attention(image_embeddings):
        attention_parameters = transformer_parameters['attention_parameters']
        axons, dentrites = cupy.array(attention_parameters[0]), cupy.array(attention_parameters[1])
        batch_size = image_embeddings.shape[0]
        num_tokens = image_embeddings.shape[1]
        total_attn_feature_size = NUM_ATTENTION_HEADS * ATTENTION_FEATURE_SIZE

        first_linear_projection = (cupy.matmul(image_embeddings, axons[0]) + dentrites[0][0, :]).reshape(batch_size, NUM_ATTENTION_HEADS, num_tokens, ATTENTION_FEATURE_SIZE)
        second_linear_projection = (cupy.matmul(image_embeddings, axons[1]) + dentrites[1][0, :]).reshape(batch_size, NUM_ATTENTION_HEADS, num_tokens, ATTENTION_FEATURE_SIZE)
        third_linear_projection = (cupy.matmul(image_embeddings, axons[2]) + dentrites[2][0, :]).reshape(batch_size, NUM_ATTENTION_HEADS, num_tokens, ATTENTION_FEATURE_SIZE)
        # attention scores -> batch | attention heads | patches | patches
        attention_scores = (cupy.matmul(first_linear_projection, second_linear_projection.transpose(0, 1, 3, 2))) / math.sqrt(ATTENTION_FEATURE_SIZE)
        # attention scores as probabilities
        attention_axons = softmax(attention_scores)
        # image_patches_context -> batch | patches | attention heads | attention feature size
        image_patches_context = cupy.matmul(attention_axons, third_linear_projection).reshape(batch_size, num_tokens, NUM_ATTENTION_HEADS, ATTENTION_FEATURE_SIZE)
        # batch | patches | attention heads * attention feature size
        attention_output = image_patches_context.reshape(batch_size, num_tokens, total_attn_feature_size)
        attention_projections = np.stack([first_linear_projection, second_linear_projection, third_linear_projection], axis=0)
        return attention_output, attention_axons, attention_projections 

    def encoder_mlp(attention_output):
        input_for_layer = attention_output
        encoder_mlp_parameters = transformer_parameters['encoder_mlp_parameters']
        axons, dentrites = cupy.array(encoder_mlp_parameters[0]), cupy.array(encoder_mlp_parameters[1])
        encoder_mlp_activations = [attention_output]
        for each in range(len(MLP_ARCHITECTURE)-1):
            input_size = MLP_ARCHITECTURE[each]
            output_size = MLP_ARCHITECTURE[each+1]
            use_activation_function = each == 0
            if use_activation_function: input_for_layer = relu(cupy.matmul(input_for_layer, axons[each, :input_size, :output_size]) + dentrites[each, 0, :output_size])
            else: input_for_layer = cupy.matmul(input_for_layer, axons[each, :input_size, :output_size]) + dentrites[each, 0, :output_size] 
            encoder_mlp_activations.append(input_for_layer)
        return input_for_layer, encoder_mlp_activations

    def encoder_layer(image_embeddings):
        # Multi-Head-Attention
        attention_output, attention_axons, attention_projections = multi_head_attention(image_embeddings)
        # Residual connection from image embeddings to attention output
        attention_residual_connection = attention_output + image_embeddings
        # Multi-Layer-Perceptron
        mlp_output, mlp_activations = encoder_mlp(attention_residual_connection)
        # Residual connection from attention residual connection to mlp output
        mlp_residual_connection = mlp_output + attention_residual_connection
        return mlp_residual_connection, attention_axons, attention_projections, image_embeddings, mlp_activations

    def encoder_forward(image_embeddings):
        encoder_layers_embeddings = []
        encoder_layers_attention_axons = []
        encoder_layers_mlp_activations = []
        encoder_layers_attention_projections = []
        encoder_input = image_embeddings
        for _ in range(NUM_LAYERS):
            encoder_input, attention_axons, attention_projections, embeddings, mlp_activations = encoder_layer(encoder_input)
            encoder_layers_embeddings.append(embeddings)
            encoder_layers_attention_axons.append(attention_axons)
            encoder_layers_attention_projections.append(attention_projections)
            encoder_layers_mlp_activations.append(mlp_activations)

        transformer_model_activations['encoder_input_embeddings'] = encoder_layers_embeddings
        transformer_model_activations['attention_projections'] = encoder_layers_attention_projections
        transformer_model_activations['attention_axons'] = encoder_layers_attention_axons
        transformer_model_activations['encoder_mlp_activations'] = encoder_layers_mlp_activations
        return encoder_input 

    def multi_layer_perceptron(encoder_output):
        mlp_parameters = transformer_parameters['mlp_parameters']
        mlp_activations = [encoder_output]
        layer_input = encoder_output
        for each in range(len(MLP_ARCHITECTURE)-1):
            use_activation_function = each == 0
            axons, dentrites = mlp_parameters[each]
            if use_activation_function: layer_input = relu(cupy.matmul(layer_input, cupy.array(axons)) + cupy.array(dentrites))
            else: layer_input = cupy.matmul(layer_input, cupy.array(axons)) + cupy.array(dentrites)
            mlp_activations.append(layer_input)
        transformer_model_activations['mlp_activations'] = mlp_activations
        return layer_input

    def model_output(mlp_output):
        output_parameters = transformer_parameters['output_parameters']
        axons, dentrites = cupy.array(output_parameters[0]), cupy.array(output_parameters[1])
        output_prediction = cupy.matmul(mlp_output, axons) + dentrites
        transformer_model_activations['model_output_previous_activation'] = mlp_output
        return output_prediction

    def model_forward(batched_image):
        embeddings = image_embeddings(batched_image)
        encoder_output = encoder_forward(embeddings)
        mlp_output = multi_layer_perceptron(encoder_output)
        output_activation = model_output(mlp_output)
        return output_activation, transformer_model_activations

    return model_forward
