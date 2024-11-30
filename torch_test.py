import torch
from torch.utils.data import DataLoader
from datasets.dataset import IAMDataset
from Model.configurations import IMAGE_SIZE
from torch_model.transformer import Transformer
from torch_based_utils import torch_model_runner

def runner():
    EPOCHS = 10
    LEARNING_RATE = 0.001
    MODEL = Transformer()
    OPTIMIZER = torch.optim.AdamW
    DATASET = IAMDataset('./datasets/Dataword', 'words.txt', IMAGE_SIZE, patch_width=16)
    TRAINING_LOADER = DataLoader(dataset=DATASET, batch_size=128, shuffle=True)
    torch_model_runner(model=MODEL, training_loader=TRAINING_LOADER, optimizer=OPTIMIZER, learning_rate=LEARNING_RATE, epochs=EPOCHS)
runner()
