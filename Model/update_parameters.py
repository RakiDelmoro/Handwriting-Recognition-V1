import cupy
from Model.configurations import NUM_LAYERS

def update_model_parameters(learning_rate, model_parameters, model_activations, model_stresses):

    def model_layer_output_update():
        activation = model_activations['model_output_previous_activation']
        layer_axons= cupy.array(model_parameters['output_parameters'][0])
        layer_dentrites = cupy.array(model_parameters['output_parameters'][1])
        stress = model_stresses['output_layer_stress']
        parameters_nudge = learning_rate * cupy.mean(cupy.matmul(activation.transpose(0, 2, 1), stress), axis=0)
        layer_axons -= parameters_nudge
        layer_dentrites -= cupy.mean(cupy.sum(stress, axis=1), axis=0)

    def mlp_layer_update():
        mlp_previous_activations = model_activations['mlp_activations'][:-1]
        mlp_layers_parameters = model_parameters['mlp_parameters']
        mlp_layers_stress = model_stresses['mlp_layers_stress']
        for each in range(len(mlp_layers_parameters)):
            layer_axons = cupy.array(mlp_layers_parameters[-(each+1)][0])
            layer_dentrites = cupy.array(mlp_layers_parameters[-(each+1)][1])
            previous_activation = mlp_previous_activations[-(each+1)]
            layer_stress = mlp_layers_stress[each]
            parameters_nudge = learning_rate * cupy.mean(cupy.matmul(previous_activation.transpose(0, 2, 1), layer_stress), axis=0)
            layer_axons -= parameters_nudge
            layer_dentrites -= cupy.mean(cupy.sum(layer_stress, axis=1), axis=0)

    def encoder_mlp_layer_update(mlp_layers_stress, mlp_previous_activations):
        previous_activations = mlp_previous_activations[:-1]
        mlp_parameters = model_parameters['encoder_mlp_parameters']
        for each in range(len(mlp_parameters)):
            layer_axons = cupy.array(mlp_parameters[-(each+1)][0])
            layer_dentrites = cupy.array(mlp_parameters[-each+1][1])
            previous_activation = previous_activations[-(each+1)]
            layer_stress = mlp_layers_stress[each]
            parameters_nudge = learning_rate * cupy.mean(cupy.matmul(previous_activation.transpose(0, 2, 1), layer_stress), axis=0)
            layer_axons -= parameters_nudge
            layer_dentrites -= cupy.mean(cupy.sum(layer_stress, axis=1), axis=0)

    def attention_layer_update(attentions_stress, input_embeddings):
        attention_parameters = model_parameters['attention_parameters']
        for each in range(len(attention_parameters)):
            layer_axons = cupy.array(attention_parameters[each][0])
            layer_dentrites = cupy.array(attention_parameters[each][1])
            previous_activation = input_embeddings
            layer_stress = attentions_stress[-(each+1)]
            parameters_nudge = learning_rate * cupy.mean(cupy.matmul(previous_activation.transpose(0, 2, 1), layer_stress), axis=0)
            layer_axons -= parameters_nudge
            layer_dentrites -= cupy.mean(cupy.sum(layer_stress, axis=1), axis=0)

    def encoder_parameters_update():
        encoder_mlp_activations = model_activations['encoder_mlp_activations']
        encoder_input_embeddings = model_activations['encoder_input_embeddings']
        encoder_mlp_stress = model_stresses['encoder_mlp_layers_stress']
        attentions_stress = model_stresses['encoder_mha_stress']
        for each in range(NUM_LAYERS):
            each_encoder_mlp_stress = encoder_mlp_stress[each]
            each_encoder_mha_stress = attentions_stress[each]
            mlp_activations = encoder_mlp_activations[each]
            embeddings = encoder_input_embeddings[each]
            encoder_mlp_layer_update(each_encoder_mlp_stress, mlp_activations)
            attention_layer_update(each_encoder_mha_stress, embeddings)

    def input_embeddings_update():
        activation = cupy.array(model_activations['input_previous_activations'])
        input_embeddings_parameters = model_parameters['image_embeddings_parameters']
        input_embeddings_stress = model_stresses['input_embeddings_stress']
        # update linear activations
        axons, dentrites = [cupy.array(input_embeddings_parameters[0][0]), cupy.array(input_embeddings_parameters[0][1])]
        parameters_nudge = learning_rate * cupy.mean(cupy.matmul(activation.transpose(0, 2, 1), input_embeddings_stress[:, 1:-1, :]), axis=0)
        axons -= parameters_nudge
        dentrites -= cupy.mean(cupy.sum(input_embeddings_stress, axis=1), axis=0)
        
        # update learnable tokens
        cls_token = cupy.array(input_embeddings_parameters[1])
        dstl_token = cupy.array(input_embeddings_parameters[2])
        position_embeddings = cupy.array(input_embeddings_parameters[3])
        
        position_embeddings -= cupy.mean(cupy.sum(input_embeddings_stress[:, 1:-1, :], axis=1), axis=0)
        cls_token -= cupy.mean(cupy.sum(input_embeddings_stress[:, 1, :], axis=1), axis=0)
        dstl_token -= cupy.mean(cupy.sum(input_embeddings_stress[:, -1, :], axis=1), axis=0)

    model_layer_output_update()
    mlp_layer_update()
    encoder_parameters_update()
    input_embeddings_update()
