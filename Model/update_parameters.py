import cupy
from Model.configurations import NUM_LAYERS

def update_model_parameters(learning_rate, model_parameters, model_activations, model_stresses):

    def model_layer_output_update():
        activation = model_activations['model_output_previous_activation']
        layer_output_parameters = [cupy.array(model_parameters['output_parameters'][0]), cupy.array(model_parameters['output_parameters'][1])]
        stress = model_stresses['output_layer_stress']
        parameters_nudge = learning_rate * cupy.mean(cupy.matmul(activation.transpose(0, 2, 1), stress), axis=0)
        layer_output_parameters[0] -= parameters_nudge
        layer_output_parameters[1] -= cupy.mean(cupy.sum(stress, axis=1), axis=0)

    def mlp_layer_update():
        mlp_previous_activations = model_activations['mlp_previous_activations']
        mlp_layers_parameters = model_parameters['mlp_parameters']
        mlp_layers_stress = model_stresses['mlp_layers_stress']
        for each in range(len(mlp_layers_parameters)):
            axons, dentrites = [cupy.array(mlp_layers_parameters[each][0]), cupy.array(mlp_layers_parameters[each][1])]
            previous_activation = cupy.array(mlp_previous_activations[each])
            layer_stress = cupy.array(mlp_layers_stress[each])
            parameters_nudge = learning_rate * cupy.mean(cupy.matmul(previous_activation.transpose(0, 2, 1), layer_stress), axis=0)
            axons -= parameters_nudge
            dentrites -= cupy.mean(cupy.sum(layer_stress, axis=1), axis=0)

    def encoder_mlp_layer_update(mlp_parameters, mlp_layers_stress):
        encoder_mlp_previous_activations = model_activations['encoder_mlp_previous_activations']
        for each in range(len(mlp_parameters)):
            axons, dentrites = [cupy.array(mlp_parameters[each][0]), cupy.array(mlp_parameters[each][1])]
            previous_activation = cupy.array(encoder_mlp_previous_activations[each])
            layer_stress = cupy.array(mlp_layers_stress[each])
            parameters_nudge = learning_rate * cupy.mean(cupy.matmul(previous_activation.transpose(0, 2, 1), layer_stress))
            axons -= parameters_nudge
            dentrites -= cupy.mean(cupy.sum(layer_stress, axis=1), axis=0)

    def attention_layer_update(attention_parameters, attentions_stress):
        previous_activations = model_activations['mha_previous_activations']
        for each in range(len(attention_parameters)):
            axons, dentrites = [cupy.array(attention_parameters[each][0]), cupy.array(attention_parameters[each][1])]
            previous_activation = cupy.array(previous_activations[each])
            layer_stress = cupy.array(attentions_stress[each])
            parameters_nudge = learning_rate * cupy.mean(cupy.matmul(previous_activation.transpose(0, 2, 1), layer_stress))
            axons -= parameters_nudge
            dentrites -= cupy.mean(cupy.sum(layer_stress, axis=1), axis=0)

    def encoder_parameters_update():
        encoder_layers_mlp_parameters = model_parameters['encoder_parameters'][1]
        encoder_layers_mha_parameters = model_parameters['encoder_parameters'][0]
        encoder_mlp_stress = model_stresses['encoder_layers_stress'][0]
        attentions_stress = model_stresses['encoder_layers_stress'][1]
        for each in range(NUM_LAYERS):
            each_encoder_mlp_parameters = encoder_layers_mlp_parameters[each]
            each_encoder_mha_parameters = encoder_layers_mha_parameters[each]
            each_encoder_mlp_stress = encoder_mlp_stress[each]
            each_encoder_mha_stress = attentions_stress[each]
            encoder_mlp_layer_update(each_encoder_mlp_parameters, each_encoder_mlp_stress)
            attention_layer_update(each_encoder_mha_parameters, each_encoder_mha_stress)

    def input_embeddings_update():
        activation = model_activations['input_previous_activations']
        input_embeddings_parameters = model_parameters['image_embeddings_parameters']
        input_embeddings_stress = model_stresses['input_embeddings_stress']
        # update linear activations
        axons, dentrites = [cupy.array(input_embeddings_parameters[0][0]), cupy.array(input_embeddings_parameters[0][1])]
        parameters_nudge = learning_rate * cupy.mean(cupy.matmul(cupy.array(activation).transpose(0, 2, 1), input_embeddings_stress[:, 1:-1, :]))
        axons -= parameters_nudge
        dentrites -= cupy.mean(cupy.sum(input_embeddings_stress, axis=1), axis=0)
        
        # update learnable tokens
        cls_token = cupy.array(input_embeddings_parameters[1])
        dstl_token = cupy.array(input_embeddings_parameters[2])
        position_embeddings = cupy.array(input_embeddings_parameters[-1])
        
        position_embeddings -= cupy.mean(cupy.sum(input_embeddings_stress[:, 1:-1, :], axis=1), axis=0)
        cls_token -= cupy.mean(cupy.sum(input_embeddings_stress[:, 1, :], axis=1), axis=0)
        dstl_token -= cupy.mean(cupy.sum(input_embeddings_stress[:, -1, :], axis=1), axis=0)

    model_layer_output_update()
    mlp_layer_update()
    encoder_parameters_update()
    input_embeddings_update()

    return model_parameters