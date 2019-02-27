import dataloader

import torch
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.FiveCrop(330),
                lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]),
                lambda norms: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])(norm) for norm in norms])])

dataset = dataloader.ImageNetDataset('data/imagespart', 'data.csv',
                                     crop_size=330,
                                     transform=transform)

dataset_loader = DataLoader(dataset,
                            batch_size=4,
                            shuffle=True,
                            num_workers=4)

device = torch.device("cuda")
model = models.resnet18(pretrained=True).to(device)
model.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
model.eval()

correct = 0
with torch.no_grad():
    for _, batch in enumerate(dataset_loader):
        batched_fives = batch['image']
        labels = batch['label'].to(device)

        batch_size, num_crops, c, h, w = batched_fives.size()
        
        # flatten over batch and five crops
        stacked_fives = batched_fives.view(-1, c, h, w).to(device)
     
        result = model(stacked_fives)
        result_avg = result.view(batch_size, num_crops, -1).mean(1) # avg over crops
        pred = result_avg.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
    
print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(dataset_loader.dataset), 
                                                       100. * correct / len(dataset_loader.dataset)))   