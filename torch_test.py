import torch
from datasets.iam_dataset import iam_dataset, iam_dataloader
from torch_model.transformer import ImageEmbeddings, MultiHeadAttention, EncoderMultiLayerPerceptron, Layer, EncoderLayer, MultiLayerPerceptron, Transformer
from Model.configurations import PATCH_SIZE, NUM_PATCHES, IMAGE_SIZE


# dataset = iam_dataset('./datasets/Dataword', 'words.txt', IMAGE_SIZE, patch_width=16)
# dataloader = iam_dataloader(dataset)

x = torch.randn(128, IMAGE_SIZE[0], IMAGE_SIZE[1])
y = torch.randint(low=0, high=83, size=(128, 20))
dataloader = [[x, y]]

loss = Transformer().training_layers(dataloader)
print(loss)