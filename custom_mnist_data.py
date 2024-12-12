import cupy
import torch
import Levenshtein
from tqdm import tqdm
from features import GREEN, RED, RESET
from datasets.utils import ints_to_characters
from Model.utils import cross_entropy_loss
from Model.backpropagation import backpropagation
from Model.update_parameters import update_model_parameters
from Model.configurations import DEVICE, NETWORK_FEATURE_SIZE, MLP_RATIO, NUMBER_OF_CLASSES, MLP_DEPTH
from Model.parameters_initalization import transformer_parameters_initializer

def model_runner(model, training_loader, validation_loader, optimizer, learning_rate, epochs):
    transformer_parameters = transformer_parameters_initializer(NETWORK_FEATURE_SIZE, MLP_DEPTH, MLP_RATIO, 10)

    def training(parameters):
        per_batch_stress = []
        training_loop = tqdm(training_loader, total=len(training_loader), leave=False)
        for image_array, expected_array in training_loop:
            model_prediction, model_activations, attention_projections, attentions_axons = model(parameters)(image_array)
            stress, layer_stress = cross_entropy_loss(model_prediction, expected_array)
            model_layers_stresses = backpropagation(layer_stress, attention_projections, attentions_axons, parameters)
            parameters = update_model_parameters(learning_rate, parameters, model_activations, model_layers_stresses)
            per_batch_stress.append(stress.item())

        return cupy.mean(cupy.array(per_batch_stress))

    def validation(parameters):
        per_batch_accuracy = []
        wrong_samples_indices = []
        correct_samples_indices = []
        model_predictions = []
        expected_model_prediction = []
        for input_image_batch, expected_batch in validation_loader:
            input_image_batch = input_image_batch.to(DEVICE)
            expected_batch = expected_batch.to(DEVICE)
            model_output = model.forward(input_image_batch)
            batch_accuracy = (expected_batch == (model_output).argmax(-1)).float().mean()
            correct_indices_in_a_batch = torch.where(expected_batch == model_output.argmax(-1))[0]
            wrong_indices_in_a_batch = torch.where(~(expected_batch == model_output.argmax(-1)))[0]

            per_batch_accuracy.append(batch_accuracy.item())
            correct_samples_indices.append(correct_indices_in_a_batch)
            wrong_samples_indices.append(wrong_indices_in_a_batch)
            model_predictions.append(cupy.array(model_output.argmax(-1).cpu().numpy()))
            expected_model_prediction.append(cupy.array(expected_batch.cpu().numpy()))
        model_accuracy = cupy.mean(cupy.array(per_batch_accuracy))
        correct_samples = cupy.concatenate(correct_samples_indices)[list(range(0,len(correct_samples_indices)))]
        wrong_samples = cupy.concatenate(wrong_samples_indices)[list(range(0,len(wrong_samples_indices)))]
        model_prediction = cupy.concatenate(model_predictions)
        model_expected_prediction = cupy.concatenate(expected_model_prediction)
        print(f"{GREEN}Model Correct Predictions{RESET}")
        for indices in correct_samples: print(f"Digit Image is: {GREEN}{model_expected_prediction[indices].item()}{RESET} Model Prediction: {GREEN}{model_prediction[indices].item()}{RESET}")
        print(f"{RED}Model Wrong Predictions{RESET}")
        for indices in wrong_samples: print(f"Digit Image is: {RED}{model_expected_prediction[indices].item()}{RESET} Model Predictions: {RED}{model_prediction[indices].item()}{RESET}")
        return model_accuracy

    for each in range(epochs):
        average_stress = training(transformer_parameters)
        accuracy = validation()
        print(f'EPOCH: {each+1} LOSS: {average_stress} ACCURACY: {accuracy}')
    