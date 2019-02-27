import os
import numpy as np
import pandas as pd
from PIL import Image

from torchvision import transforms
from torch.utils.data.dataset import Dataset

# the dataset should load the images & labels as pairs
class ImageNetDataset(Dataset):
    def __init__(self, directory, csv_path, crop_size, transform=None):
        """
        Args:
            csv_path (string): path to preprocessed csv with pairs 'path,label'
            transform (optional): pytorch transforms for transforms and tensor conversion
        """
        self.to_tensor = transforms.ToTensor()
        self.directory = directory
        self.image_label_pairs = pd.read_csv(csv_path, header=None)
        self.transform = transform
        self.crop_size = crop_size

    def _load_image(self, filename):
        """
        This function loads an image into memory when you give it
        the path of the image. Able to handle grayscale images.
        """
        img = Image.open(filename)
        img.load()
        img = np.array(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)
            img = np.repeat(img, 3, 2)
        return Image.fromarray(img)

    def _resize(self, image, size):
        """
        This function resizes a PIL image so that the smaller side is
        equal to 'size'.
        """
        w, h = image.size
        scale = size / min(w,h)
        new_size = (int(np.ceil(scale * w)),
                    int(np.ceil(scale * h)))
        return image.resize(new_size)

    def __len__(self):
        return len(self.image_label_pairs)

    def __getitem__(self, idx):
        img_name = os.path.join(self.directory,
                                self.image_label_pairs.iloc[idx, 0])
        image = self._load_image(img_name)
        label = self.image_label_pairs.iloc[idx, 1]
        if self.transform is not None:
            image = self._resize(image, self.crop_size)
            image = self.transform(image)
        sample = {'image': image, 'label': label}
        return sample
