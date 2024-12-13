import cupy
import torch
from tqdm import tqdm
from features import GREEN, RED, RESET
from Model.configurations import DEVICE

def model_runner(model, training_loader, validation_loader, optimizer, learning_rate, epochs):
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

    def validation():
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
        accuracy = validation()
        # save_model_weights(each+1, average_stress, optim_state)
        print(f'EPOCH: {each+1} LOSS: {average_stress} ACCURACY: {accuracy}')
    