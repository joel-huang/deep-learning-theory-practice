import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

import torchvision.models as models
from torchvision import transforms

from PIL import Image
import numpy as np
import pandas as pd
import os

class FlowerDataset(Dataset):
    def __init__(self, image_dir, image_paths, label_file, transform=None):
        self.image_dir = image_dir
        self.labels = np.load(label_file)
        self.image_label_pairs = self._load_paths(image_paths)
        self.transform = transform
        
    def train_val_test_split(self, train_ratio, val_ratio):
        dataset_length = len(self.image_label_pairs)
        train_length = int(train_ratio * dataset_length)
        val_length = int(val_ratio * dataset_length)
        test_length = len(self) - train_length - val_length
        splits = [train_length, val_length, test_length]
        return random_split(self, splits)
        
    def _load_paths(self, file_path):
        """
        params:  file_path, a path pointing to where the image paths are stored.
        returns: dictionary with keys 'full_image_path', and values 'label'
        """
        split_set = {}
        with open(file_path) as f:
            lines = f.readlines()
            num_lines = len(lines)
            assert(num_lines == len(self.labels))
            for line_num in range(num_lines):
                full_image_path = os.path.join(self.image_dir, lines[line_num].strip('\n'))
                split_set[full_image_path] = self.labels[line_num]
        return pd.DataFrame.from_dict(split_set, orient='index')
        
    def _load_image(self, image_path):
        img = Image.open(image_path)
        img.load()
        img = np.array(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)
            img = np.repeat(img, 3, 2)
        return Image.fromarray(img)
        
    def __len__(self):
        return len(self.image_label_pairs)
    
    def __getitem__(self, idx):
        # apply transforms
        image_path = self.image_label_pairs.index[idx]
        image = self._load_image(image_path)
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        return {'image': image,
                'label': label}
    
transform = transforms.Compose([transforms.CenterCrop(200), transforms.ToTensor()])
dataset = FlowerDataset('data', 'image_paths.txt', 'labels.npy', transform=transform)
train_set, val_set, test_set = dataset.train_val_test_split(0.7, 0.1)


train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=32, shuffle=True, num_workers=4)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_losses = []
    for idx, batch in enumerate(train_loader):
        data, target = batch['image'].to(device), batch['label'].long().to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = CE(output, target)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        print('Epoch: {}, Samples: {}/{}, Loss: {}'.format(epoch, idx*batch_size,
                                                           len(train_loader)*batch_size,
                                                           loss.item()))
    train_loss = torch.mean(torch.tensor(train_losses))
    print('\nEpoch: {}'.format(epoch))
    print('Training set: Average loss: {:.4f}'.format(train_loss))
    
    return train_loss

def validate(model, device, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    
    with torch.no_grad():
        for _, batch in enumerate(val_loader):
            data, target = batch['image'].to(device), batch['label'].long().to(device)
            output = model(data)
            
            # compute the batch loss
            batch_loss = CE(output, target).item()
            val_loss += batch_loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # divide by the number of batches of batch size 32
    # get the average validation over all bins
    val_loss /= len(val_loader)
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    
    return val_loss

DATA_DIRECTORY = 'data/'
use_cuda = 1
batch_size = 32
num_epochs = 50
learning_rate = 1e-3

device = torch.device("cuda" if use_cuda else "cpu")
model = models.resnet18(num_classes=102).to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
CE = nn.CrossEntropyLoss()

train_losses = []
val_losses = []

for epoch in range(1, num_epochs + 1):
    train_loss = train(model, device, train_loader, optimizer, epoch)
    val_loss = validate(model, device, val_loader)

    if (len(val_losses) > 0) and (val_loss < min(val_losses)):
        torch.save(model.state_dict(), "best_model.pt")
        print("Saving model (epoch {}) with lowest validation loss: {}"
              .format(epoch, val_loss))

    train_losses.append(train_loss)
    val_losses.append(val_loss)

print("Training and validation complete.")

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(12,5))
epoch_list = np.arange(1, num_epochs+1)
plt.xticks(epoch_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(epoch_list, train_losses, label="Training loss")
plt.plot(epoch_list, val_losses, label="Validation loss")
plt.legend(loc='upper right')
plt.show()

model.eval()

correct = 0
with torch.no_grad():
    for _, batch in enumerate(test_loader):
        data = batch['image'].to(device)
        labels = batch['label'].long().to(device)
        result = model(data)
        pred = result.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
    
print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset), 
                                                       100. * correct / len(test_loader.dataset)))