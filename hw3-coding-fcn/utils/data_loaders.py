import torch

from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import Dataset, DataLoader

def get_data_loaders(batch_size, directory):
    train_data = FashionMNIST(directory, train=True, transform=transforms.ToTensor(), download=True)
    test_data = FashionMNIST(directory, train=False, transform=transforms.ToTensor(), download=True)

    train_data_loader = torch.utils.data.DataLoader(train_data,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=4)
    
    test_data_loader = torch.utils.data.DataLoader(test_data,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4)
    return train_data_loader, test_data_loader