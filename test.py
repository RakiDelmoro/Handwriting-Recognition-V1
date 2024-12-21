import cupy
from Model.configurations import NUMBER_OF_CLASSES, PADDING_IDX, IMAGE_SIZE, PATCH_WIDTH
from Model.transformer import transformer_model
from datasets.iam_dataset import iam_dataloader, iam_dataset

# dataset = iam_dataset('./datasets/Dataword', 'words.txt', (64, 512), patch_width=16)
# dataloader = iam_dataloader(dataset)
x = cupy.random.randn(128, 10, 768)
y = cupy.random.randint(low=0, high=83, size=(128, 10))
model = transformer_model(768, conv_depth=5, patch_window=(3,3), patches_ratio=2, mlp_depth=3, mlp_ratio=2, number_of_classes=10)
model(x)
