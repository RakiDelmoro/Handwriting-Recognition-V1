import cupy
import torch
import math
from cupy import asnumpy
from torch.nn.functional import ctc_loss, softmax
from Model.backpropagation import backpropagation
from neural_network_layers import convolution_neurons
from Model.parameters_initalization import transformer_parameters_initializer
from Model.configurations import NUM_ATTENTION_HEADS, ATTENTION_FEATURE_SIZE

def transformer_model(network_feature_size, conv_depth, patch_window, patches_ratio, mlp_depth, mlp_ratio, number_of_classes):
    transformer_parameters = transformer_parameters_initializer(network_feature_size, conv_depth, patch_window, patches_ratio, mlp_depth, mlp_ratio, output_class=number_of_classes)
    convolution_parameters = transformer_parameters['conv_parameters']
    mha_parameters = transformer_parameters['attn_parameters']
    encoder_mlp_parameters = transformer_parameters['enc_mlp_parameters']

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

    def encoder_mlp(attention_output, parameters=None):
        input_for_layer = attention_output
        input_feature_size = attention_output.shape[-1]
        output_feature_size = input_feature_size*mlp_ratio
        if parameters is None:
            layer_1_axons, layer_1_dentrites = axons_and_dentrites_initialization(input_feature_size, output_feature_size)
            layer_2_axons, layer_2_dentrites = axons_and_dentrites_initialization(output_feature_size, input_feature_size)
        else:
            layer_1_axons, layer_1_dentrites = parameters[0]
            layer_2_axons, layer_2_dentrites = parameters[1]
        layer_1_activations = cupy.matmul(attention_output, layer_1_axons) + layer_1_dentrites
        layer_2_activations = cupy.matmul(layer_1_activations, layer_2_axons) + layer_2_dentrites
        mlp_activations = [asnumpy(input_for_layer), asnumpy(layer_1_activations), asnumpy(layer_2_activations)]
        mlp_parameters = [[asnumpy(layer_1_axons), asnumpy(layer_1_dentrites)], [asnumpy(layer_2_axons), asnumpy(layer_2_dentrites)]]
        return layer_2_activations, mlp_activations, mlp_parameters

    def encoder_layer(image_embeddings, attention_parameters=None, mlp_parameters=None):
        # Multi-Head-Attention
        attention_output, attention_activations, attn_parameters = multi_head_attention(image_embeddings, attention_parameters)
        # Residual connection from image embeddings to attention output
        attention_residual_connection = attention_output + image_embeddings
        # Multi-Layer-Perceptron
        mlp_output, mlp_activations, mlp_params = encoder_mlp(attention_residual_connection, mlp_parameters)
        # Residual connection from attention residual connection to mlp output
        mlp_residual_connection = mlp_output + attention_residual_connection # Second residual connection
        encoder_layer_activations = [asnumpy(attention_residual_connection), asnumpy(mlp_residual_connection)]
        return mlp_residual_connection, encoder_layer_activations, attention_activations, mlp_activations, attn_parameters, mlp_params

    def encoder_forward(image_embeddings):
        attention_activation_inside_layers = []
        mlp_activation_inside_layers = []
        encoder_activations = []
        encoder_parameters = []
        encoder_input = image_embeddings
        for _ in range(num_layers):
            encoder_input, encoder_layer_activation, attention_activations, mlp_activations, attn_params, mlp_params = encoder_layer(encoder_input)
            encoder_activations.append(encoder_layer_activation)
            attention_activation_inside_layers.append(attention_activations)
            mlp_activation_inside_layers.append(mlp_activations)
            encoder_parameters.append([attn_params, mlp_params])
        return encoder_input, encoder_activations, attention_activation_inside_layers, mlp_activation_inside_layers, encoder_parameters

    def multi_layer_perceptron(encoder_output, parameters=None):
        input_feature_size = encoder_output.shape[-1]
        output_feature_size = input_feature_size*mlp_ratio
        if parameters is None:
            layer_1_axons, layer_1_dentrites = axons_and_dentrites_initialization(input_feature_size, output_feature_size)
            layer_2_axons, layer_2_dentrites = axons_and_dentrites_initialization(output_feature_size, input_feature_size)
        else:
            layer_1_axons, layer_1_dentrites = parameters[0]
            layer_2_axons, layer_2_dentrites = parameters[1]
        layer_1_activations = cupy.matmul(encoder_output, layer_1_axons) + layer_1_dentrites
        layer_2_activations = cupy.matmul(layer_1_activations, layer_2_axons) + layer_2_dentrites
        mlp_activations = [asnumpy(layer_1_activations), asnumpy(layer_2_activations)]
        mlp_parameters = [[asnumpy(layer_1_axons), asnumpy(layer_1_dentrites)], [asnumpy(layer_2_axons), asnumpy(layer_2_dentrites)]]
        return layer_2_activations, mlp_activations, mlp_parameters

    def model_output(mlp_output, parameters=None):
        input_feature_size = mlp_output.shape[-1]
        output_feature_size = number_of_classes
        if parameters is None:
            output_axons, output_dentrites = axons_and_dentrites_initialization(input_feature_size, output_feature_size)
        else:
            output_axons, output_dentrites = parameters
        model_output = log_softmax(cupy.matmul(mlp_output, output_axons) + output_dentrites)
        model_output_parameters = [asnumpy(output_axons), asnumpy(output_dentrites)]
        return model_output, model_output_parameters

    def model_forward(image_patches):
        batch_size = image_patches.shape[0]
        num_image_patches = image_patches.shape[1]
        image_patches = image_patches.reshape(batch_size, num_image_patches, -1)
        # Instead of using CNN feature extraction we use simple image embeddings for simplicity.
        image_embeddings, embeddings_parameters = convolution_layers(image_patches)
        encoder_output, encoder_activations, attention_activations, encoder_mlp_activations, encoder_parameters = encoder_forward(image_embeddings)
        mlp_output, mlp_activations, mlp_parameters = multi_layer_perceptron(encoder_output)
        model_prediction, output_layer_parameters = model_output(mlp_output)
        return model_prediction, image_embeddings, encoder_activations, attention_activations, encoder_mlp_activations, mlp_activations, embeddings_parameters, encoder_parameters, mlp_parameters, output_layer_parameters

    def calculate_network_stress(model_prediction, expected_prediction):
        # Use TORCH for calculation loss
        batch_size = model_prediction.shape[0]
        num_image_patches = model_prediction.shape[1]
        idx_prediction = model_prediction.shape[-1]
        # num patches | batch | number of classes
        model_prediction = torch.tensor(model_prediction.reshape(num_image_patches, batch_size, idx_prediction), device='cuda', requires_grad=True)
        expected_prediction = torch.tensor(expected_prediction, device='cuda')
        image_patches_length = torch.full(size=(batch_size,), fill_value=num_image_patches, dtype=torch.long)
        word_token_length = unpadded_length_tokens(expected_prediction)
        model_loss = ctc_loss(model_prediction, expected_prediction, image_patches_length, word_token_length)
        # get the model prediction gradients
        model_loss.backward()
        last_layer_stress = cupy.array(model_prediction.grad)
        return cupy.array(model_loss.item()), last_layer_stress

    def update_network_parameters(): pass
    #TODO: Create a function for updating parameters

    def training_model(training_loader):
        for batched_image_patch, expected_word in training_loader:
            batched_image_patch = cupy.array(batched_image_patch)
            expected_word = cupy.array(expected_word)
            model_outputs = model_forward(batched_image_patch)
            model_prediction = model_outputs[0]
            # model activations
            image_embeddings, encoder_activations, attention_activation, encoder_mlp_activations, mlp_activations = model_outputs[1:-4]
            # model parameters
            image_embeddings_parameters, encoder_parameters, mlp_parameters, output_layer_parameters = model_outputs[6:]
            # Calculate network stress
            network_stress, last_layer_stress = calculate_network_stress(model_prediction, expected_word)
            mlp_stress, encoder_stress, embeddings_stress = backpropagation(last_layer_stress, output_layer_parameters, mlp_parameters, encoder_parameters, image_embeddings_parameters)
            # Update network parameters
            # update_network_parameters(image_embeddings_parameters, encoder_parameters, mlp_parametesr, output_layer_parameters)
    
    return multi_head_attention
