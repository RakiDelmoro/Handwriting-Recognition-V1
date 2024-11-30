from datasets.iam_dataset import iam_dataset
from torch.utils.data import Dataset

class IAMDataset(Dataset):
    def __init__(self, folder, txt_file, image_size, patch_width):
        super().__init__()
        self.dataset = iam_dataset(folder=folder, txt_file=txt_file, image_size=image_size, patch_width=patch_width)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        return self.dataset[index]