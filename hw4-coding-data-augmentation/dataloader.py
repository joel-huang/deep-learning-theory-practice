import getimagenetclasses
from torch.utils.data.dataset import Dataset
from torch.utils.data import RandomSampler

class ImageNetDataset(Dataset):
    def __init__(self, image_paths, label_paths):
        self.path = path

class ImageNetDataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        if shuffle:
            sampler = RandomSampler(dataset)

