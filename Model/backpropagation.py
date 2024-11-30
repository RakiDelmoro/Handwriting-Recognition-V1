import cupy
from torch import tensor
from torch.nn.functional import softmax
from Model.configurations import ATTENTION_FEATURE_SIZE, NUM_ATTENTION_HEADS, NETWORK_FEATURE_SIZE, NUM_PATCHES

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

    def softmax_layer_stress(layer_stress):
        # We use torch for getting the derivative of the softmax
        layer_stress = tensor(layer_stress, device='cuda', requires_grad=True)
        loss = softmax(layer_stress, dim=-1).sum()
        # Get the gradients
        loss.backward()
        return cupy.array(layer_stress.grad)

    def calculate_attention_stress(layer_stress_propagated, encoder_parameters):
        attention_parameters = encoder_parameters[0]
        # attn output layer stress
        attn_output_layer_axons, _ = attention_parameters[-1]
        attn_output_layer_stress = cupy.matmul(layer_stress_propagated, cupy.array(attn_output_layer_axons.transpose()))
        # Value projection stress
        attn_score_in_probabilities = attention_parameters[-2]
        # batch | attention heads | patches | attention feature size
        image_patches_context_stress = attn_output_layer_stress.reshape(attn_output_layer_stress.shape[0], NUM_ATTENTION_HEADS, NUM_PATCHES, ATTENTION_FEATURE_SIZE)
        # batch | attention heads | attention feature size | patches
        value_projection_stress = cupy.matmul(image_patches_context_stress.transpose(0,1,3,2), cupy.array(attn_score_in_probabilities).transpose(0,1,3,2)).transpose(0, 1, 3, 2)
        # batch | attention heads | patches | attention feature size
        value_projection = attention_parameters[-3]
        # batch | attention heads | patches | patches
        attention_weighting_stress = cupy.matmul(cupy.array(value_projection), image_patches_context_stress.transpose(0,1,3,2))
        attention_scores_stress = cupy.matmul(cupy.array(attn_score_in_probabilities), attention_weighting_stress)
        # query, key projections -> batch | attention heads | patches | attention feature size
        key_projection = cupy.array(attention_parameters[-4])
        query_projection = cupy.array(attention_parameters[-5])
        # batch | attention heads | patches | patches
        key_projection_layer_stress = softmax_layer_stress(attention_scores_stress)
        query_projection_layer_stress = softmax_layer_stress(attention_scores_stress)
        # Shape of stress to aggregate
        batch_size = query_projection.shape[0]
        patches = query_projection.shape[2]
        value_stress_to_aggregate = value_projection_stress.reshape(batch_size, patches, -1)
        key_stress_to_aggregate = cupy.matmul(key_projection_layer_stress, query_projection).reshape(batch_size, patches, -1)
        query_stress_to_aggregate = cupy.matmul(query_projection_layer_stress, key_projection).reshape(batch_size, patches, -1)
        # query, key and value parameters
        value_projection_axons = cupy.array(attention_parameters[-6][0])
        key_projection_axons = cupy.array(attention_parameters[-7][0])
        query_projection_axons = cupy.array(attention_parameters[-8][0])
        input_embeddings_stress = (cupy.matmul(value_stress_to_aggregate, value_projection_axons.transpose()) + cupy.matmul(key_stress_to_aggregate, key_projection_axons.transpose()) + cupy.matmul(query_stress_to_aggregate, query_projection_axons.transpose()))
        attention_projection_stresses = [cupy.asnumpy(attn_output_layer_stress), cupy.asnumpy(value_stress_to_aggregate), cupy.asnumpy(key_stress_to_aggregate), cupy.asnumpy(query_stress_to_aggregate), cupy.asnumpy(input_embeddings_stress)]
        return input_embeddings_stress, attention_projection_stresses

    def calculate_encoder_layers_stress(layer_stress_propagated):
        layer_stress = layer_stress_propagated
        encoder_layers_stress = []
        for each_encoder_layer in range(len(encoder_parameters)):
            # tuple of mlp parameters and attention layer parameters
            encoder_layer_parameters = encoder_parameters[-(each_encoder_layer+1)]
            mlp_stress_propagated, mlp_layers_stress = calculate_mlp_encoder_stress(layer_stress, encoder_layer_parameters)
            attention_stress_propagated, attention_projections_stress = calculate_attention_stress(mlp_stress_propagated, encoder_layer_parameters)
            encoder_stress = [mlp_layers_stress, attention_projections_stress]
            encoder_layers_stress.append(encoder_stress)
        return attention_stress_propagated, encoder_layers_stress

    def calculate_input_embeddings_stress(layer_stress_propagated):
        input_embeddings_axons = image_embeddings_parameters[0]
        return cupy.matmul(layer_stress_propagated, cupy.array(input_embeddings_axons.transpose()))

    layers_propagated = propagate_network_stress(last_layer_stress)
    mlp_stress_propagated, mlp_layers_stress = calculate_mlp_stress(layers_propagated)
    encoder_stress_propagated, encoder_stress = calculate_encoder_layers_stress(mlp_stress_propagated)
    embeddings_stress = calculate_input_embeddings_stress(encoder_stress_propagated)

    return mlp_layers_stress, encoder_stress, embeddings_stress
