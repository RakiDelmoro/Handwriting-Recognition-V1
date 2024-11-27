from datasets.utils import CHARS, PAD_TOKEN, characters_to_ints

IMAGE_SIZE = 128, 512
PATCH_SIZE = IMAGE_SIZE[0], IMAGE_SIZE[0]
ATTENTION_FEATURE_SIZE = 768
NUM_ATTENTION_HEADS = 16
NUM_LAYERS = 24
MLP_FEATURE_SIZE = ATTENTION_FEATURE_SIZE*2
NUMBER_OF_CLASSES = len(CHARS)
PADDING_IDX = characters_to_ints(PAD_TOKEN)