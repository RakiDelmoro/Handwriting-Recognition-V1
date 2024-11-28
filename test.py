import cupy
from Model.configurations import NUMBER_OF_CLASSES, PADDING_IDX
from Model.transformer import transformer_model
from datasets.iam_dataset import iam_dataloader, iam_dataset

# dataset = iam_dataset('./datasets/Dataword', 'words.txt', (64, 512), patch_width=16)
# dataloader = iam_dataloader(dataset)
x = cupy.random.randn(128, 32, 64, 16)
y = cupy.random.randint(low=0, high=83, size=(128, 10))
model = transformer_model(network_feature_size=256, num_attn_heads=8, num_layers=2, attention_feature_size=128, mlp_ratio=4, number_of_classes=NUMBER_OF_CLASSES, padding_token=PADDING_IDX)

model([[x, y]])

# for image_batch, target_word_batch in dataloader:
#     model(dataloader)
