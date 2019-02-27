import dataloader

import torch
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
dataset = dataloader.ImageNetDataset('data/imagespart', 'data.csv',
                                     crop_size=224,
                                     transform=transform)
dataset_loader = DataLoader(dataset,
                            batch_size=4,
                            shuffle=True,
                            num_workers=4)

device = torch.device("cuda")
model = models.resnet18(pretrained=True).to(device)
model.eval()

correct = 0
with torch.no_grad():
    for _, batch in enumerate(dataset_loader):
        images, labels = batch['image'], batch['label']
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
    
print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(dataset_loader.dataset), 
                                                       100. * correct / len(dataset_loader.dataset))) 