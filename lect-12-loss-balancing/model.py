import torch
import torch.nn as nn
import torch.nn.functional as F

class LogReg(nn.Module):
    def __init__(self):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(2, 1)
        
    def forward(self, x):
        x = self.fc(x)
        return torch.sigmoid(x)