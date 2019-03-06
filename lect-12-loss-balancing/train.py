import model
import dataset
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import BCELoss

num_epochs = 10
learning_rate = .1
batch_size = 128

ibd = dataset.BalancedImbalancedDataset('samplestr.txt')
train_loader = DataLoader(ibd, batch_size=batch_size, shuffle=True, num_workers=4)

ibd_test = dataset.BalancedImbalancedDataset('sampleste.txt')
test_loader = DataLoader(ibd_test, batch_size=batch_size, shuffle=True, num_workers=4)

ibd.plot_data()
ibd_test.plot_data()

model = model.LogReg()
loss_function = BCELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

train_loss = []

def train(epoch):
    model.train()
    for batch_id, batch in enumerate(train_loader):
        X, y = batch[0], batch[1]
        optimizer.zero_grad()
        y_pred = model(X)
        batch_loss = loss_function(y_pred, y)
        batch_loss.backward()
        optimizer.step()
        train_loss.append(batch_loss)
        print("Epoch: {} | Batch {}/{} | Loss: {}".format(epoch,
            batch_id * batch_size, len(train_loader.dataset), batch_loss))

def test():
    model.eval()
    test_loss, correct, true_positive = 0, 0, 0 
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            X, y = batch[0], batch[1]
            y_pred = model(X)
            batch_loss = loss_function(y_pred, y).item()
            test_loss += batch_loss
            pred = torch.zeros(y_pred.shape)
            pred[np.where(y_pred > 0.5)] = 1.0
            correct += pred.eq(y.view_as(pred)).sum().item()
            # true_positive += 
    test_loss /= len(test_loader)
    print("Test accuracy: {}/{} | Test Loss: {}".format(correct, len(test_loader.dataset), test_loss))

for epoch in range(num_epochs):
    train(epoch)

test()