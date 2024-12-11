import cupy

def update_model_parameters(learning_rate, model_parameters, model_activations, model_stresses):

    def model_layer_output_update():
        activation = model_activations['model_output_previous_activation']
        layer_output_parameters = model_parameters['output_parameters']
        stress = model_stresses['output_layer_stress']
        parameters_nudge = learning_rate * cupy.mean(cupy.matmul(activation.transpose(0, 2, 1), stress), axis=0)
        layer_output_parameters[0] -= parameters_nudge
        layer_output_parameters[1] -= cupy.mean(cupy.sum(stress, axis=1), axis=0)

    def mlp_layer_update():
        mlp_previous_activations = model_activations['mlp_previous_activations']
        mlp_layers_parameters = model_parameters['mlp_parameters']
        mlp_layers_stress = model_stresses['mlp_layers_stress']
        for each in range(len(mlp_layers_parameters)):
            axons, dentrites = mlp_layers_parameters[each]
            previous_activation = mlp_previous_activations[each]
            layer_stress = mlp_layers_stress[each]
            parameters_nudge = learning_rate * cupy.mean(cupy.matmul(previous_activation.transpose(0, 2, 1), layer_stress))
            axons -= parameters_nudge
            dentrites -= cupy.mean(cupy.sum(layer_stress, axis=1), axis=0)

    def encoder_mlp_layer_update():
        encoder_mlp_previous_activations = model_activations['encoder_mlp_previous_activations']
        encoder_mlp_parameters = model_parameters['encoder_parameters'][1]
        encoder_mlp_stress = model_stresses['encoder_layers_stress'][0]
        for each in range(len(encoder_mlp_parameters)):
            axons, dentrites = encoder_mlp_parameters[each]
            previous_activation = encoder_mlp_previous_activations[each]
            layer_stress = encoder_mlp_stress[each]
            parameters_nudge = learning_rate * cupy.mean(cupy.matmul(previous_activation.transpose(0, 2, 1), layer_stress))
            axons -= parameters_nudge
            dentrites -= cupy.mean(cupy.sum(layer_stress, axis=1), axis=0)

    def attention_layer_update():
        attention_parameters = model_parameters['encoder_parameters'][0]
        attention_stress = model_stresses['encoder_layers_stress'][1]
        for each in range(len(attention_parameters)):
            axons, dentrites = attention_parameters
            previous_activation = attention_parameters[each]
            layer_stress = attention_stress[each]
            parameters_nudge = learning_rate * cupy.mean(cupy.matmul(previous_activation.transpose(0, 2, 1), layer_stress))
            axons -= parameters_nudge
            dentrites -= cupy.mean(cupy.sum(layer_stress, axis=1), axis=0)

    def input_embeddings_update():
        activation = model_activations['input_embeddings']
        input_embeddings_parameters = model_parameters['input_embeddings']
        input_embeddings_stress = model_stresses['input_embeddings_stress']
        # update linear activations
        axons, dentrites = input_embeddings_parameters[0]
        parameters_nudge = learning_rate * cupy.mean(cupy.matmul(activation.transpose(0, 2, 1), input_embeddings_stress))
        axons -= parameters_nudge
        dentrites -= cupy.mean(cupy.sum(input_embeddings_stress, axis=1), axis=0)
        # update learnable position embeddings
        position_embeddings = input_embeddings_parameters[-1]     
        position_embeddings -= cupy.mean(cupy.sum(input_embeddings_stress, axis=1), axis=0)

    model_layer_output_update()
    mlp_layer_update()
    encoder_mlp_layer_update()
    attention_layer_update()
    input_embeddings_update()

    return model_parameters