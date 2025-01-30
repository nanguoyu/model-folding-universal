from torch import nn
from typing import Any
import torch
from typing import Any
from functools import wraps


__all__ = ['MLP', 'mlp']

def clean_mlp_kwargs(func):
    @wraps(func)
    def wrapper(**kwargs):
        valid_kwargs = ['num_classes', 'wider_factor', 'n_channels', 'weights']
        extra_kwargs = {k: kwargs[k] for k in kwargs if k not in valid_kwargs}
        if extra_kwargs:
            print(f"Removed unsupported kwargs: {list(extra_kwargs.keys())}")
            kwargs = {k: kwargs[k] for k in kwargs if k in valid_kwargs}
        return func(**kwargs)
    return wrapper

class MLP(nn.Module):
    def __init__(self, num_classes: int = 10, wider_factor: int = 1, n_channels: int = 3):
        super(MLP, self).__init__()
        self.n_channels = n_channels
        self.num_classes = num_classes
        self.fc1 = nn.Linear(n_channels*32*32, 512*wider_factor)
        self.bn1 = nn.BatchNorm1d(512*wider_factor)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512*wider_factor, 256*wider_factor)
        self.bn2 = nn.BatchNorm1d(256*wider_factor)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256*wider_factor, 128*wider_factor)
        self.bn3 = nn.BatchNorm1d(128*wider_factor)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128*wider_factor, num_classes)

    def forward(self, x):
        x = x.reshape(-1, 32*32*self.n_channels)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

@clean_mlp_kwargs
def mlp(**kwargs: Any) -> MLP:
    pre_kwargs = {k: kwargs[k] for k in kwargs if k != 'weights'}
    model = MLP(**pre_kwargs)
    print("Create mlp: initialized")
    if 'weights' in kwargs:
        model.load_state_dict(torch.load(kwargs['weights'], map_location='cpu'))
        print(f"Load mlp weights from {kwargs['weights']}")
    return model