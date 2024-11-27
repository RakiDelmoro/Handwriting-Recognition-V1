import cupy
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
        # batch size | patches | image patched feature size
        return image_embeddings, axons, dentrites

    def multi_head_attention(input_tokens, parameters=None):
        batch_size = input_tokens.shape[0]
        num_tokens = input_tokens.shape[1]
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
        image_patches_query = (cupy.dot(input_tokens, axons_for_query) + dentrites_for_query).reshape(batch_size, num_attn_heads, num_tokens, attention_feature_size)
        image_patches_key = (cupy.dot(input_tokens, axons_for_key) + dentrites_for_key).reshape(batch_size, num_attn_heads, num_tokens, attention_feature_size)
        image_patches_value = (cupy.dot(input_tokens, axons_for_value) + dentrites_for_value).reshape(batch_size, num_attn_heads, num_tokens, attention_feature_size)
        # attention scores -> batch | attention heads | tokens | tokens
        attention_scores = cupy.matmul(image_patches_query, image_patches_key.transpose(0, 1, 3, 2))
        # attention scores as probabilities
        attentions_probabilities = softmax(attention_scores)
        # image_patches_context -> batch | patches | attention heads | attention feature size
        image_patches_context = cupy.matmul(attentions_probabilities, image_patches_value).reshape(batch_size, num_tokens, num_attn_heads, attention_feature_size)
        # batch | tokens | attention heads * attention feature size
        attention_output = image_patches_context.reshape(batch_size, num_tokens, total_attn_feature_size)
        # batch | tokens | network feature size
        attention_output = cupy.dot(attention_output, output_attention_axons) + output_attnetion_dentrites
        return attention_output, [[axons_for_query, dentrites_for_query], [axons_for_key, dentrites_for_key], [axons_for_value, dentrites_for_value], [output_attention_axons, output_attnetion_dentrites]]

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
        encoder_mlp_activations = [input_for_layer, layer_1_activations, layer_2_activations]
        return encoder_mlp_activations, [[layer_1_axons, layer_1_dentrites], [layer_2_axons, layer_2_dentrites]]

    def encoder_layer(tokens_embeddings, attention_parameters=None, mlp_parameters=None):
        self_attention_output, self_attention_params = multi_head_attention(tokens_embeddings, attention_parameters)
        attention_residual_connection = self_attention_output + tokens_embeddings # First residual connection
        mlp_activations, mlp_params = encoder_mlp(attention_residual_connection, mlp_parameters)
        mlp_residual_connection = mlp_activations[-1] + self_attention_output # Second residual connection
        encoder_layer_activations = [tokens_embeddings, self_attention_output, attention_residual_connection, mlp_activations, mlp_residual_connection]
        return encoder_layer_activations, self_attention_params, mlp_params

    def encoder_forward(image_embeddings):
        encoder_activations = []
        encoder_parameters = []
        encoder_input = image_embeddings
        for _ in range(num_layers):
            activations, attention_parameters, mlp_parameters = encoder_layer(encoder_input)
            encoder_input = activations[-1]
            encoder_activations.append(activations)
            encoder_parameters.append([attention_parameters, mlp_parameters])
        return encoder_activations, encoder_parameters
    
    def mlp(encoder_output, parameters=None):
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
        mlp_activations = [layer_1_activations, layer_2_activations]
        return mlp_activations
    
    def model_output(mlp_output, parameters=None):
        input_feature_size = mlp_output.shape[-1]
        output_feature_size = number_of_classes
        if parameters is None:
            output_axons, output_dentrites = axons_and_dentrites_initialization(input_feature_size, output_feature_size)
        else:
            output_axons, output_dentrites = parameters
        model_output = cupy.dot(mlp_output, output_axons) + output_dentrites
        return model_output

    def model_forward(image_patches, word_tokens):
        image_embeddings, axons, dentrites = image_patches_embeddings(image_patches)
        encoder_activations, encoder_parameters = encoder_forward(image_embeddings)
        encoder_output = encoder_activations[-1][-1]
        mlp_activations = mlp(encoder_output)
        mlp_output = mlp_activations[-1]
        model_prediction = model_output(mlp_output)
        return model_prediction

    return model_forward
