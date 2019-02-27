import dataloader

import torch
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader

import re
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

class DenseNetModified(models.DenseNet):
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out

def densenet121(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNetModified(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet121'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

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
model = densenet121(pretrained=True).to(device)
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