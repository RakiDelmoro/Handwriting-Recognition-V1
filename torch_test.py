import torch
from datasets.utils import split_data
from torch.utils.data import DataLoader
from datasets.dataset import IAMDataset
from Model.configurations import IMAGE_SIZE
from torch_model.transformer import Transformer
from torch_based_utils import torch_model_runner

def runner():
    EPOCHS = 50
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    MODEL = Transformer()
    OPTIMIZER = torch.optim.AdamW
    DATASET = IAMDataset('./datasets/Dataword', 'words.txt', IMAGE_SIZE, patch_width=16)
    TRAINING_DATASET, VALIDATION_DATASET = split_data(dataset=DATASET, ratio=0.8)
    TRAINING_LOADER = DataLoader(dataset=TRAINING_DATASET, batch_size=BATCH_SIZE, shuffle=True)
    VALIDATION_LOADER = DataLoader(dataset=VALIDATION_DATASET, batch_size=BATCH_SIZE, shuffle=True)
    # TRAINING_LOADER = [[torch.randn(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1]), torch.randint(low=0, high=83, size=(BATCH_SIZE, 32))]] # Test samples
    torch_model_runner(model=MODEL, training_loader=TRAINING_LOADER, validation_loader=VALIDATION_LOADER, optimizer=OPTIMIZER, learning_rate=LEARNING_RATE, epochs=EPOCHS)

runner()
