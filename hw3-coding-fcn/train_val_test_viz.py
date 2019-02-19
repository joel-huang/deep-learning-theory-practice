import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.model import FCN
from utils.data_loaders import get_data_loaders
from utils.scoring import get_classwise_accuracy
from utils.visualization import plot_training_validation_loss
from utils.visualization import plot_classwise_accuracies

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item()) 
    
    train_loss = torch.mean(torch.tensor(train_losses))
    print('Training set: Average loss: {:.4f}'.format(train_loss))
    
    return train_loss

def validate(model, device, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # sum up all the batch losses
            val_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # get the average validation loss
    val_loss /= len(val_loader.dataset)
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    
    return val_loss

def test(model, device, test_loader):
    model.eval()
    
    num_classes = 10
    outputs = []
    classes = []
    confusion_matrix = torch.zeros(num_classes, num_classes)
    
    with torch.no_grad():
        for _, (input_batch, class_list) in enumerate(test_loader):
            input_batch, class_list = input_batch.to(device), class_list.to(device)
            output = model(input_batch)
            _, preds = torch.max(output, 1)
            for t, p in zip(class_list.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
           
    classwise_accuracy = get_classwise_accuracy(confusion_matrix)
    best_accuracy = torch.mean(classwise_accuracy)
    print('Test set: Best accuracy: {}'.format(best_accuracy))
    
    return classwise_accuracy

def main():

    parser = argparse.ArgumentParser(description='Train, validate, test, visualize')
    parser.add_argument('--dir', default='data/', help='Directory of the dataset')
    parser.add_argument('--cuda', default=1, help='Use CUDA, 1 or 0')
    parser.add_argument('--batch_size', default=16, help='Batch size')
    parser.add_argument('--num_epochs', default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', default=1e-2, help='Learning rate')
    args = parser.parse_args()

    DATA_DIRECTORY = args.dir
    use_cuda = int(args.cuda)
    batch_size = int(args.batch_size)
    num_epochs = int(args.num_epochs)
    learning_rate = float(args.learning_rate)

    train_loader, test_loader = get_data_loaders(batch_size, DATA_DIRECTORY)
    device = torch.device("cuda" if use_cuda else "cpu")
    model = FCN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        val_loss = validate(model, device, test_loader) # use test as val (wrong)
        
        if (len(val_losses) > 0) and (val_loss < min(val_losses)):
            torch.save(model.state_dict(), "fashion_mnist_fcn.pt")
            print("Saving model (epoch {}) with lowest validation loss: {}"
                  .format(epoch, val_loss))
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
    print("Training and validation complete.")

    print("Loading model for inference.")
    model.load_state_dict(torch.load("fashion_mnist_fcn.pt"))

    print("Running inference.")
    classwise_accuracies = test(model, device, test_loader)

    plot_training_validation_loss(num_epochs, train_losses, val_losses)
    plot_classwise_accuracies(classwise_accuracies)

if __name__ == '__main__':
    main()
    