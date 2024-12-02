import cupy
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
            stress = model.get_stress_and_update_parameters(model_prediction, expected_array, optimizer, learning_rate)
            per_batch_stress.append(stress.item())
        return cupy.mean(cupy.array(per_batch_stress))
    
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
    
        for each in per_batch_str_similarities:
            counter = 0
            string_distance, model_and_expected_str = next(iter(each.items()))
            if string_distance in lowest_similarties:
                print(f"Predicted: {model_and_expected_str[0]}, Expected: {model_and_expected_str[1]}")
                counter+=1
            if counter >= len(lowest_similarties): break
        
        for each in per_batch_str_similarities:
            counter = 0
            string_distance, model_and_expected_str = next(iter(each.items()))
            if string_distance in highest_similarities:
                print(f"Predicted: {model_and_expected_str[0]} Expected: {model_and_expected_str[1]}")
                counter+=1
            if counter >= len(highest_similarities): break

    for each in range(epochs):
        average_stress = training()
        validation()
        print(f'EPOCH: {each+1} LOSS: {average_stress}')
