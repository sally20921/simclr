import json

from PIL import Image
from torch.utils.data import Dataset


class INaturalist2019Dataset(Dataset):

    def __init__(self, json_file_path, dataset_dir_path, transform=None):

        self.images = []

        if not json_file_path.exists():
            raise FileNotFoundError('json_file_path does not exist: %s' % json_file_path)

        with open(json_file_path, "r") as read_file:
            self.images = json.load(read_file)

        self.dataset_dir_path = dataset_dir_path
        self.transform = transform
        self.classes = range(1010)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_file_key = list(self.images[idx])[0]
        image_file_path = self.dataset_dir_path.joinpath(image_file_key)

        image = Image.open(image_file_path).convert('RGB')

        category = self.images[idx][image_file_key]

        if self.transform is not None:
            image = self.transform(image)

        return image, category
