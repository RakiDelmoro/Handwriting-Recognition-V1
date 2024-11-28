import cupy
from Model.utils import softmax

def backpropagation(last_layer_stress, output_layer_parameters, mlp_parameters, encoder_parameters, image_embeddings_parameters):

    def propagate_network_stress(last_layer_stress):
        output_layer_axons, _ = output_layer_parameters
        # transpose last layer stress - num_patches | batch | feature size -> batch | num_patches | feature size
        layer_stress_propagated = cupy.matmul(last_layer_stress.transpose(1, 0, 2), cupy.array(output_layer_axons.transpose()))
        return layer_stress_propagated

    def calculate_mlp_stress(layer_stress_propagated):
        layer_stress = layer_stress_propagated
        mlp_layers_stress = []
        for layer_idx in range(len(mlp_parameters)):
            layer_axons, _ = mlp_parameters[-(layer_idx+1)]
            layer_stress = cupy.matmul(layer_stress, cupy.array(layer_axons.transpose()))
            mlp_layers_stress.append(cupy.asnumpy(layer_stress))
        return layer_stress, mlp_layers_stress

    def calculate_mlp_encoder_stress(layer_stress_propagated, encoder_parameters):
        layer_stress = layer_stress_propagated
        encoder_mlp_layers_stress = []
        mlp_parameters = encoder_parameters[1]
        for layer_idx in range(len(mlp_parameters)):
            layer_axons, _ = mlp_parameters[-(layer_idx+1)]
            layer_stress = cupy.matmul(layer_stress, cupy.array(layer_axons.transpose()))
            encoder_mlp_layers_stress.append(cupy.asnumpy(layer_stress))
        return layer_stress, encoder_mlp_layers_stress

    def calculate_attention_stress(layer_stress_propagated, encoder_parameters):
        layer_stress = layer_stress_propagated
        attention_parameters = encoder_parameters[0]
        # attn output layer stress
        attn_output_layer_axons, _ = attention_parameters[-1]
        attn_output_layer_stress = cupy.matmul(layer_stress_propagated, cupy.array(attn_output_layer_axons.transpose()))
        # Value projection stress
        attn_score_in_probabilities = attention_parameters[-2]
        # batch | attention heads | patches | attention feature size
        #TODO: refactor avoid hard coded
        attn_output_layer_stress_propagated = attn_output_layer_stress.reshape(attn_output_layer_stress.shape[0], 8, 32, 128)
        # bhsd -> batch | attention heads | patches | attention feature size
        # bhns -> batch | attention heads | patches | patches
        value_projection_layer_stress = cupy.matmul(cupy.array(attn_score_in_probabilities), attn_output_layer_stress_propagated)
        value_projection = attention_parameters[-3]
        attention_score_stress = cupy.matmul(attn_output_layer_stress_propagated.transpose(0, 1, 3, 2), cupy.array(value_projection))
        key_projection_layer_stress = softmax(cupy.array(attn_score_in_probabilities), attention_score_stress, return_derivative=True)
        query_projection_layer_stress = softmax(cupy.array(attn_score_in_probabilities), attention_score_stress, return_derivative=True)
        #TODO: Implement a function to calculate the stress for Query, Key, Value projections

    def calculate_encoder_layers_stress(layer_stress_propagated):
        layer_stress = layer_stress_propagated
        for each_encoder_layer in range(len(encoder_parameters)):
            # tuple of mlp parameters and attention layer parameters
            encoder_layer_parameters = encoder_parameters[-(each_encoder_layer+1)]
            mlp_stress_propagated, mlp_layers_stress = calculate_mlp_encoder_stress(layer_stress, encoder_layer_parameters)
            attention_stress_propagated, attention_projections_stress = calculate_attention_stress(mlp_stress_propagated, encoder_layer_parameters)

    layers_propagated = propagate_network_stress(last_layer_stress)
    mlp_stress_propagated, mlp_layers_stress = calculate_mlp_stress(layers_propagated)
    calculate_encoder_layers_stress(mlp_stress_propagated)