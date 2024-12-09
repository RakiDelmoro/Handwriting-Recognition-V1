import cupy
import torch
import Levenshtein
from tqdm import tqdm
from datasets.utils import ints_to_characters
from Model.configurations import DEVICE

def torch_model_runner(model, training_loader, validation_loader, optimizer, learning_rate, epochs):

    def training():
        per_batch_stress = []
        training_loop = tqdm(training_loader, total=len(training_loader), leave=False)
        for image_array, expected_array in training_loop:
            image_array = image_array.to(DEVICE)
            expected_array = expected_array.to(DEVICE)
            model_prediction = model.forward(image_array)
            stress, optim_state = model.get_stress_and_update_parameters(model_prediction, expected_array, optimizer, learning_rate)
            per_batch_stress.append(stress.item())
        return cupy.mean(cupy.array(per_batch_stress)), optim_state

    def batch_calculate_str_distance(model_prediction_in_batch, expected_prediction_in_batch):
        similarities_corresponding_model_and_expected_str = []
        batch_iterator = model_prediction_in_batch.shape[0]
        for each in range(batch_iterator):
            model_prediction, expected_prediction = model_prediction_in_batch[each], expected_prediction_in_batch[each]
            model_prediction_as_indices = model.prediction_as_indices(model_prediction)
            model_prediction_as_str = model.prediction_as_str(model_prediction_as_indices)
            expected_prediction_as_str = ints_to_characters(expected_prediction)
            distance_error = Levenshtein.distance(model_prediction_as_str, expected_prediction_as_str)
            max_length = max(len(model_prediction_as_str), len(expected_prediction_as_str))
            similarity = float(max_length - distance_error) / float(max_length)
            similarities_corresponding_model_and_expected_str.append({similarity: (model_prediction_as_str, expected_prediction_as_str)})
        return similarities_corresponding_model_and_expected_str

    def validation():
        per_batch_str_similarities = []
        validation_loop = tqdm(validation_loader, total=len(validation_loader), leave=False)
        for image_array, expected_array in validation_loop:
            image_array = image_array.to(DEVICE)
            expected_array = expected_array.to(DEVICE)
            model_prediction = model.forward(image_array)
            # Batch calculate str distance return tuple: str similarites, model_and_expected_as_str
            batch_similarities = batch_calculate_str_distance(model_prediction, expected_array)
            per_batch_str_similarities.extend(batch_similarities)
        list_of_similarities = [similarity for each_dict in per_batch_str_similarities for similarity in each_dict.keys()]
        highest_similarities = sorted(list_of_similarities, reverse=True)[:5]
        lowest_similarties = sorted(list_of_similarities)[:5]
        low_similarities_counter = 0
        for each in per_batch_str_similarities:
            string_distance, model_and_expected_str = next(iter(each.items()))
            if string_distance in lowest_similarties:
                print(f"Predicted: {model_and_expected_str[0]}, Expected: {model_and_expected_str[1]}")
                low_similarities_counter += 1
            if low_similarities_counter >= len(lowest_similarties): break
        highest_similarities_counter = 0
        for each in per_batch_str_similarities:
            string_distance, model_and_expected_str = next(iter(each.items()))
            if string_distance in highest_similarities:
                print(f"Predicted: {model_and_expected_str[0]} Expected: {model_and_expected_str[1]}")
                highest_similarities_counter += 1
            if highest_similarities_counter >= len(highest_similarities): break

    def save_model_weights(epoch, model_stress, optim_state):
        torch.save({'epoch': epoch,
                    'model state': model.state_dict(),
                    'optimizer state': optim_state,
                    'loss': model_stress}, 'model.pt')
        print('Saved checkpoint!')

    def load_model_weights():
        checkpoint = torch.load('model.pt')
        model.load_state_dict(checkpoint['model state'])
        optimizer.load_state_dict(checkpoint['optimizer state'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f'Model path: model.pt epoch: {epoch} loss: {loss}')

    for each in range(epochs):
        average_stress, optim_state = training()
        validation()
        save_model_weights(each+1, average_stress, optim_state)
        print(f'EPOCH: {each+1} LOSS: {average_stress}')
