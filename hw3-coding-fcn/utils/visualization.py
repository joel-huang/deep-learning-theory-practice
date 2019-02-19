import numpy as np
import matplotlib.pyplot as plt

def plot_training_validation_loss(num_epochs, train_losses, test_losses):
    epoch_list = np.arange(1, num_epochs+1)
    plt.xticks(epoch_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epoch_list, train_losses, label="Training loss")
    plt.plot(epoch_list, test_losses, label="Validation loss")
    plt.legend(loc='upper right')
    plt.show()
    
def plot_classwise_accuracies(classwise_accuracies):
    indices = np.arange(len(classwise_accuracies))
    accuracies = dict(sorted(zip(classwise_accuracies, indices)))
    sorted_x = [str(x) for x in accuracies.values()]
    sorted_accuracies = list(accuracies.keys())
    
    fig = plt.figure(figsize=(20,5))
    plt.subplot(121)
    plt.xticks(indices)
    plt.ylabel("Accuracy", size='x-large')
    plt.xlabel("Classes", size='x-large')
    plt.bar(indices, list(classwise_accuracies))
    plt.subplot(122)
    plt.ylabel("Accuracy", size='x-large')
    plt.xlabel("Classes (Sorted by accuracy)", size='x-large')
    plt.bar(sorted_x, sorted_accuracies)
    plt.show()