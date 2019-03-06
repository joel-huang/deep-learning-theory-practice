import PIL
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


def main():
    lr = 1
    target_class = 949

    img = PIL.Image.open("./mrhero.jpg")
    resnet18 = models.resnet18(pretrained=True).eval()
    
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    weights = torch.zeros(1000)
    weights[target_class] = 1.0

    loss_func = nn.CrossEntropyLoss(weight=weights)
    
    img = transform(img)
    orig = unnormalize(img, means, stds)

    transforms.ToPILImage()(orig).show()
    inp = img[None]
    pred = resnet18(inp).argmax(dim=1, keepdim=True)

    while pred.item() != target_class:
        inp = torch.autograd.Variable(inp, requires_grad=True)
        output = resnet18(inp)
        pred = output.argmax(dim=1, keepdim=True)

        class_output = output[0][target_class]
        class_output.backward()

        mask = validity_mask(inp)
        inp = inp + lr * mask.float() * inp.grad.data
        print("Current result: {} | Target result: {} | Output: {}"
            .format(pred.item(), target_class, class_output))

    
    new_img = inp[0]
    new_img = unnormalize(new_img, means, stds)
    transforms.ToPILImage()(new_img).show()
    pred = resnet18(inp).argmax(dim=1, keepdim=True)
    print(pred)

    transforms.ToPILImage()(new_img - orig).show()

def validity_mask(test_array):
    test_array = inp + lr * inp.grad.data
    zero_mask = test_array[0] >= 0
    ones_mask = test_array[0] <= 1
    return ~(~zero_mask + ~ones_mask)

def unnormalize(inp, means, stds):
    out = inp.new(*inp.size())
    for channel in range(len(means)):
        out[channel,:,:] = inp[channel,:,:] * stds[channel] + means[channel]
    return out

if __name__ == '__main__':
    main()