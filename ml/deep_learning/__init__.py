import torch
from torch import nn
from torch.optim import Adam
from torch.nn import BCELoss


class BaseModel(nn.Module):
    def __init__(self, n_inputs, n_outputs, **kwargs):
        super(BaseModel, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self):
        pass