import cupy
from cupy import asnumpy
from torch import tensor
from torch.nn.functional import softmax
from Model.configurations import ATTENTION_FEATURE_SIZE, NUM_ATTENTION_HEADS, NETWORK_FEATURE_SIZE, NUM_PATCHES

def backpropagation(layer_stress, transformer_parameters):
    image_embeddings_parameters = transformer_parameters['image_embeddings_parameters']
    mha_parameters = transformer_parameters['attn_parameters']
    encoder_mlp_parameters = transformer_parameters['enc_mlp_parameters']
    mlp_parameters = transformer_parameters['mlp_parameters']
    output_parameters = transformer_parameters['output_parameters']

    def output_layer_propagate_stress(layer_stress):
        axons = cupy.array(output_parameters[0])
        layer_stress = cupy.matmul(layer_stress, axons.transpose())
        return layer_stress

    def mlp_layers_propagate_stress(layer_stress):
        mlp_layers_stress = []
        stress_propagated = layer_stress
        for each in range(len(mlp_parameters)):
            axons = cupy.array(mlp_parameters[-(each+1)][0])
            stress_propagated = cupy.matmul(stress_propagated, axons.transpose())
            mlp_layers_stress.append(asnumpy(stress_propagated))
        return stress_propagated, mlp_layers_stress

    def encoder_mlp_layers_propagate_stress(layer_stress):
        encoder_mlp_layers_stress = []
        stress_propagated = layer_stress
        for each in range(len(encoder_mlp_parameters)):
            axons = cupy.array(mlp_parameters[-(each+1)][0])
            stress_propagated = cupy.matmul(stress_propagated, axons.transpose())
            encoder_mlp_layers_stress.append(asnumpy(stress_propagated))
        return stress_propagated, encoder_mlp_layers_stress

    def attention_layer_propagate_stress(layer_stress):
        attention_axons = mha_parameters[-1]
        query_proj, key_proj, value_proj = mha_parameters[-2]
        batch, num_patches, feature_size = layer_stress.shape
        # reshape stress -> batch | patches | attention heads | feature size
        stress_propagated = layer_stress.reshape(batch, num_patches, NUM_ATTENTION_HEADS, ATTENTION_FEATURE_SIZE)
        # image_context_stress -> batch | attention heads | patches | patches
        image_context_stress = cupy.matmul(stress_propagated, attention_axons.transpose(0, 1, 3, 2))

    layer_output_stress = output_layer_propagate_stress(layer_stress)
    mlp_stress, mlp_layers_stress = mlp_layers_propagate_stress(layer_output_stress)
    enc_mlp_stress, encoder_mlp_layers_stress = encoder_mlp_layers_propagate_stress(mlp_stress)
    attention_layer_propagate_stress(enc_mlp_stress)
