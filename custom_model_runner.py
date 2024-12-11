import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from Model.transformer import transformer_model
from Model.configurations import NETWORK_FEATURE_SIZE, MLP_RATIO, BATCH_SIZE
from custom_mnist_data import model_runner
def runner():
    EPOCHS = 100
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 28
    LEARNING_RATE = 0.001
    NUMBER_OF_CLASSES = 10
    OPTIMIZER = torch.optim.AdamW
    TRANSFORM = lambda x: transforms.ToTensor()(x).reshape(IMAGE_HEIGHT, IMAGE_WIDTH).type(dtype=torch.float32)
    TARGET_TRANSFORM = lambda x: torch.tensor(x, dtype=torch.int64)
    model = transformer_model

    training_dataset = datasets.MNIST('./training-data', download=True, train=True, transform=TRANSFORM, target_transform=TARGET_TRANSFORM)
    validation_dataset = datasets.MNIST('./training-data', download=True, train=False, transform=TRANSFORM, target_transform=TARGET_TRANSFORM)
    training_dataloader = DataLoader(dataset=training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model_runner(model, training_dataloader, validation_dataloader, OPTIMIZER, LEARNING_RATE, EPOCHS)

runner()
