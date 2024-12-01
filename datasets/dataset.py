from datasets.iam_dataset import iam_dataset, image_to_array, word_to_token_array
from torch.utils.data import Dataset

class IAMDataset(Dataset):
    def __init__(self, folder, txt_file, image_size, patch_width):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.patch_width = patch_width
        self.dataset = iam_dataset(folder=folder, txt_file=txt_file, image_size=image_size, patch_width=patch_width)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        image_path = self.dataset[index][0]
        word_written_in_image = self.dataset[index][-1]
        word_as_character_tokens = word_to_token_array(word_written_in_image)
        image_array = image_to_array(folder=self.folder, image_path=image_path, size=self.image_size, patch_width=self.patch_width)
        return image_array, word_as_character_tokens
