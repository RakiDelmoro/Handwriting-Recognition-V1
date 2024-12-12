import cupy
from cupy import asnumpy
from torch import tensor
from torch.nn.functional import softmax
from Model.configurations import ATTENTION_FEATURE_SIZE, NUM_ATTENTION_HEADS, NUM_LAYERS

def backpropagation(layer_stress, attention_projections, attentions_axons, transformer_parameters):
    encoder_parameters = transformer_parameters['encoder_parameters']
    mlp_parameters = transformer_parameters['mlp_parameters']
    output_parameters = transformer_parameters['output_parameters']

    transformer_layers_stress = {}

    def output_layer_propagate_stress(layer_stress):
        transformer_layers_stress['output_layer_stress'] = layer_stress
        axons = output_parameters[0]
        layer_stress = cupy.matmul(layer_stress, axons.transpose())
        return layer_stress

    def mlp_layers_propagate_stress(layer_stress):
        mlp_layers_stress = []
        stress_propagated = layer_stress
        for each in range(len(mlp_parameters)):
            axons = mlp_parameters[-(each+1)][0]
            stress_propagated = cupy.matmul(stress_propagated, axons.transpose())
            mlp_layers_stress.append(stress_propagated)
        transformer_layers_stress['mlp_layers_stress'] = mlp_layers_stress
        return stress_propagated

    def encoder_mlp_layers_propagate_stress(layer_stress, encoder_mlp_parameters):
        encoder_mlp_layers_stress = []
        stress_propagated = layer_stress
        for each in range(len(encoder_mlp_parameters)):
            axons = cupy.array(mlp_parameters[-(each+1)][0])
            stress_propagated = cupy.matmul(stress_propagated, axons.transpose())
            encoder_mlp_layers_stress.append(stress_propagated)
        return stress_propagated, encoder_mlp_layers_stress

    def attention_layer_propagate_stress(layer_stress, mha_parameters, attention_axons):
        query_proj, key_proj, value_proj = attention_projections
        query_proj_params, key_proj_params, value_proj_params = mha_parameters
        batch, num_patches, feature_size = layer_stress.shape
        # reshape stress -> batch | patches | attention heads | feature size
        image_context_stress = layer_stress.reshape(batch, NUM_ATTENTION_HEADS, num_patches, ATTENTION_FEATURE_SIZE)
        # image_context_stress -> batch | attention heads | patches | patches
        attention_axons_stress = cupy.matmul(image_context_stress, value_proj.transpose(0, 1, 3, 2))
        # apply softmax derivative function
        attention_stress_propagate = attention_axons_stress * attention_axons * (1 - attention_axons)
        value_projection_propagated_stress = image_context_stress.reshape(batch, num_patches, feature_size)
        key_projection_propagated_stress = cupy.matmul(attention_stress_propagate, key_proj).reshape(batch, num_patches, feature_size)
        query_projection_propagated_stress = cupy.matmul(attention_stress_propagate, query_proj).reshape(batch, num_patches, feature_size)

        value_stress = cupy.matmul(value_projection_propagated_stress, value_proj_params[0].transpose())
        key_stress = cupy.matmul(key_projection_propagated_stress, key_proj_params[0].transpose())
        query_stress = cupy.matmul(query_projection_propagated_stress, query_proj_params[0].transpose())
        attention_stress = value_stress + key_stress + query_stress
        projections_stress = [value_stress, key_stress, query_stress]
        return attention_stress, projections_stress

    def backpropagate_encoder_layers(layer_stress):
        encoder_layer_mlp_layers_stress = []
        encoder_layer_mha_stress = []
        for each in range(NUM_LAYERS):
            attention_parameters = encoder_parameters[0][each]
            encoder_mlp_parameters = encoder_parameters[1][each]
            attention_axons = attentions_axons[each]
            mlp_stress_propagated, mlp_layers_stress = encoder_mlp_layers_propagate_stress(layer_stress, encoder_mlp_parameters)
            attention_stress_propagated, attention_projections_stress = attention_layer_propagate_stress(mlp_stress_propagated, attention_parameters, attention_axons)
            layer_stress = attention_stress_propagated
            encoder_layer_mlp_layers_stress.append(mlp_layers_stress)
            encoder_layer_mha_stress.append(attention_projections_stress)
        transformer_layers_stress['encoder_layers_stress'] = [encoder_layer_mlp_layers_stress, encoder_layer_mha_stress]
        return layer_stress

    layer_output_stress = output_layer_propagate_stress(layer_stress)
    mlp_stress = mlp_layers_propagate_stress(layer_output_stress)
    transformer_stress = backpropagate_encoder_layers(mlp_stress)
    transformer_layers_stress['input_embeddings_stress'] = transformer_stress
    return transformer_layers_stress
