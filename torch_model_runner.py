import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch_model.transformer_patches import Transformer
from torch_based_mnist_utils import model_runner
from torchsummary import summary

def runner():
    EPOCHS = 50
    BATCH_SIZE = 1024
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 28
    LEARNING_RATE = 0.001
    NUMBER_OF_CLASSES = 10
    OPTIMIZER = torch.optim.SGD
    TRANSFORM = lambda x: transforms.ToTensor()(x).reshape(IMAGE_HEIGHT, IMAGE_WIDTH).type(dtype=torch.float32)
    TARGET_TRANSFORM = lambda x: torch.tensor(x, dtype=torch.int64)
    model = Transformer()

    training_dataset = datasets.MNIST('./training-data', download=True, train=True, transform=TRANSFORM, target_transform=TARGET_TRANSFORM)
    validation_dataset = datasets.MNIST('./training-data', download=True, train=False, transform=TRANSFORM, target_transform=TARGET_TRANSFORM)
    training_dataloader = DataLoader(dataset=training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # TRAINING_LOADER = [[torch.randn(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH), torch.randint(low=0, high=10, size=(BATCH_SIZE,))]] # Test samples
    model_runner(model, training_dataloader, validation_dataloader, OPTIMIZER, LEARNING_RATE, EPOCHS)
    # print(summary(model, (28, 28), batch_size=1024))
    # print(sum(p.numel() for p in model.parameters()))

runner()
