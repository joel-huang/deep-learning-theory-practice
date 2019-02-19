import torch

def get_classwise_accuracy(confusion_matrix):
    classwise_accuracies = confusion_matrix.diag()/confusion_matrix.sum(1)
    return classwise_accuracies