import cupy

def torch_model_runner(model, training_loader, validation_loader, optimizer, learning_rate, epochs):

    def training():
        per_batch_stress = []
        for image_array, expected_array in training_loader:
            model_prediction = model.forward(image_array)
            stress = model.get_stress_and_update_parameters(model_prediction, expected_array, optimizer, learning_rate)
            per_batch_stress.append(stress.item())
        return cupy.mean(cupy.array(per_batch_stress))
    
    # def validation():
    #     for image_array, expected_array in validation_loader:
    #         model_prediction = model.forward(image_array)
    #     pass
    
    for each in range(epochs):
        average_stress = training()
        print(f'EPOCH: {each+1} LOSS: {average_stress}')
