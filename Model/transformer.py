import cupy
from cupy import asnumpy
from Model.utils import axons_and_dentrites_initialization, softmax, unpadded_length_tokens

def transformer_model(network_feature_size, num_attn_heads, num_layers, attention_feature_size, mlp_ratio, number_of_classes, padding_token):

    def image_patches_embeddings(image_patches, parameters=None):
        batch_size = image_patches.shape[0]
        num_patches = image_patches.shape[1]
        input_feature = image_patches.shape[-1]
        output_feature = network_feature_size
        if parameters is None: axons, dentrites = axons_and_dentrites_initialization(input_feature, output_feature)
        else: axons, dentrites = parameters
        # This array is to provide information of image patches position. Unlike RNN which process data sequentially. Transformer work on all tokens/patches simultaneuously.
        trainable_positional_embedding = cupy.zeros([batch_size, num_patches, output_feature])
        image_projection = cupy.dot(image_patches, axons) + dentrites
        image_embeddings = image_projection + trainable_positional_embedding
        image_embeddings_params = [asnumpy(axons), asnumpy(dentrites)]
        # batch size | patches | image patched feature size
        return image_embeddings, image_embeddings_params

    def multi_head_attention(image_embeddings, parameters=None):
        batch_size = image_embeddings.shape[0]
        num_tokens = image_embeddings.shape[1]
        total_attn_feature_size = num_attn_heads * attention_feature_size
        if parameters is None: 
            axons_for_query, dentrites_for_query = axons_and_dentrites_initialization(network_feature_size, total_attn_feature_size)
            axons_for_key, dentrites_for_key = axons_and_dentrites_initialization(network_feature_size, total_attn_feature_size)
            axons_for_value, dentrites_for_value = axons_and_dentrites_initialization(network_feature_size, total_attn_feature_size)
            output_attention_axons, output_attnetion_dentrites = axons_and_dentrites_initialization(total_attn_feature_size, network_feature_size)
        else:
            axons_for_query, dentrites_for_query = parameters[0]
            axons_for_key, dentrites_for_key = parameters[1]
            axons_for_value, dentrites_for_value = parameters[2]
            output_attention_axons, output_attnetion_dentrites = parameters[3]
        # batch | attention heads | tokens | attention feature size
        image_embeddings_query = (cupy.dot(image_embeddings, axons_for_query) + dentrites_for_query).reshape(batch_size, num_attn_heads, num_tokens, attention_feature_size)
        image_embeddings_key = (cupy.dot(image_embeddings, axons_for_key) + dentrites_for_key).reshape(batch_size, num_attn_heads, num_tokens, attention_feature_size)
        image_embeddings_value = (cupy.dot(image_embeddings, axons_for_value) + dentrites_for_value).reshape(batch_size, num_attn_heads, num_tokens, attention_feature_size)
        # attention scores -> batch | attention heads | tokens | tokens
        attention_scores = cupy.matmul(image_embeddings_query, image_embeddings_key.transpose(0, 1, 3, 2))
        # attention scores as probabilities
        attentions_probabilities = softmax(attention_scores)
        # image_patches_context -> batch | patches | attention heads | attention feature size
        image_patches_context = cupy.matmul(attentions_probabilities, image_embeddings_value).reshape(batch_size, num_tokens, num_attn_heads, attention_feature_size)
        # batch | tokens | attention heads * attention feature size
        attention_output = image_patches_context.reshape(batch_size, num_tokens, total_attn_feature_size)
        # batch | tokens | network feature size
        attention_output = cupy.dot(attention_output, output_attention_axons) + output_attnetion_dentrites
        attention_activations = [asnumpy(image_embeddings_query), asnumpy(image_embeddings_key), asnumpy(image_embeddings_value)]
        attention_parameters = [[asnumpy(axons_for_query), asnumpy(dentrites_for_query)], [asnumpy(axons_for_key), asnumpy(dentrites_for_key)], [asnumpy(axons_for_value), asnumpy(dentrites_for_value)], [asnumpy(output_attention_axons), asnumpy(output_attnetion_dentrites)]]
        return attention_output, attention_activations, attention_parameters

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
        layer_1_activations = cupy.dot(attention_output, layer_1_axons) + layer_1_dentrites
        layer_2_activations = cupy.dot(layer_1_activations, layer_2_axons) + layer_2_dentrites
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
        layer_1_activations = cupy.dot(encoder_output, layer_1_axons) + layer_1_dentrites
        layer_2_activations = cupy.dot(layer_1_activations, layer_2_axons) + layer_2_dentrites
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
        model_output = cupy.dot(mlp_output, output_axons) + output_dentrites
        model_output_parameters = [asnumpy(output_axons), asnumpy(output_dentrites)]
        return model_output, model_output_parameters

    def model_forward(image_patches, word_tokens):
        batch_size = image_patches.shape[0]
        num_image_patches = image_patches.shape[1]
        image_patches = image_patches.reshape(batch_size, num_image_patches, -1)
        # Instead of using CNN feature extraction we use simple image embeddings for simplicity.
        image_embeddings, embeddings_parameters = image_patches_embeddings(image_patches)
        encoder_output, encoder_activations, attention_activations, encoder_mlp_activations, encoder_parameters = encoder_forward(image_embeddings)
        mlp_output, mlp_activations, mlp_parameters = multi_layer_perceptron(encoder_output)
        model_prediction, output_layer_parameters = model_output(mlp_output)
        return model_prediction
    # TODO: Implement backpropagation, Update Model parameters
    # TODO: Find a better approach on storing model activations and model parameters
    return model_forward
