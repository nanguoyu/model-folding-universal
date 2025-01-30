from torch import nn
from typing import Any
import torch
from typing import Any
from functools import wraps

from utils import deprecated
__all__ = ['LeNet5', 'lenet5', 'LeNet5_BN', 'lenet5_bn', 'LeNet_Deep', 'lenet_deep', 'reslenet5_bn', 'ResLeNet5_BN', 'lenet11_bn', 'LeNet11_BN']


def clean_lenet_kwargs(func):
    @wraps(func)
    def wrapper(**kwargs):
        valid_kwargs = ['num_classes', 'wider_factor', 'n_channels', 'weights']
        extra_kwargs = {k: kwargs[k] for k in kwargs if k not in valid_kwargs}
        if extra_kwargs:
            print(f"Removed unsupported kwargs: {list(extra_kwargs.keys())}")
            kwargs = {k: kwargs[k] for k in kwargs if k in valid_kwargs}
        return func(**kwargs)
    return wrapper

class LeNet5(nn.Module):
    def __init__(self, num_classes=10, wider_factor=8, n_channels=1):
        if wider_factor>1:
            print(f'Created a {wider_factor}Xwider LeNet5')
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=8*wider_factor, kernel_size=5, stride=1)
        self.relu2 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(in_channels=8*wider_factor, out_channels=32*wider_factor, kernel_size=5, stride=1)
        self.relu5 = nn.ReLU()
        self.pool6 = nn.MaxPool2d(2)



        self.fl7 = nn.Flatten()
        self.fc8 = nn.Linear(in_features=wider_factor*32 * 5 * 5, out_features=256*wider_factor)
        self.relu9 = nn.ReLU()
        self.fc10 = nn.Linear(in_features=256*wider_factor, out_features=128*wider_factor)
        self.relu11 = nn.ReLU()
        self.fc12 = nn.Linear(in_features=128*wider_factor, out_features=num_classes)

        # Initialize weights
        self._init_weight()


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu2(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.relu5(x)
        x = self.pool6(x)

        x = self.fl7(x)

        x = self.fc8(x)
        x = self.relu9(x)
        x = self.fc10(x)
        x = self.relu11(x)

        x = self.fc12(x)

        return x


    def _init_weight(self):
        # Initialize weights of original layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

@clean_lenet_kwargs
def lenet5(**kwargs: Any) -> LeNet5:
    pre_kwargs = {k: kwargs[k] for k in kwargs if k != 'weights'}

    model = LeNet5(**pre_kwargs)
    if 'weights' in kwargs:
        model.load_state_dict(torch.load(kwargs['weights'], map_location='cpu'))
        print(f"Load lenet5 weights from {kwargs['weights']}")
    else:
        print("Initialized lenet5")
    return model


class LeNet5_BN(nn.Module):
    def __init__(self, num_classes=10, wider_factor=8, n_channels=1):
        if wider_factor>1:
            print(f'Created a {wider_factor}Xwider LeNet5')
        super(LeNet5_BN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=4*wider_factor, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(4*wider_factor)
        self.relu2 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(in_channels=4*wider_factor, out_channels=16*wider_factor, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(16*wider_factor)
        self.relu5 = nn.ReLU()
        self.pool6 = nn.MaxPool2d(2)



        self.fl7 = nn.Flatten()
        self.fc8 = nn.Linear(in_features=wider_factor*16 * 5 * 5, out_features=256*wider_factor)
        self.relu9 = nn.ReLU()
        self.fc10 = nn.Linear(in_features=256*wider_factor, out_features=128*wider_factor)
        self.relu11 = nn.ReLU()
        self.fc12 = nn.Linear(in_features=128*wider_factor, out_features=num_classes)

        # Initialize weights
        self._init_weight()


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn2(x)
        x = self.relu5(x)
        x = self.pool6(x)

        x = self.fl7(x)

        x = self.fc8(x)
        x = self.relu9(x)
        x = self.fc10(x)
        x = self.relu11(x)

        x = self.fc12(x)

        return x


    def _init_weight(self):
        # Initialize weights of original layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

@deprecated
def lenet5_bn(**kwargs: Any) -> LeNet5:
    model = LeNet5_BN(**kwargs)
    return model


class LeNet_Deep(nn.Module):
    def __init__(self, num_classes=10, wider_factor = 1, n_channels=1): 
        self.wider_factor = wider_factor
        print(f'Created a LeNet_Deep')
        super(LeNet_Deep, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )

        

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=wider_factor*512, out_features=256*wider_factor),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=256*wider_factor, out_features=128*wider_factor),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=128*wider_factor, out_features=num_classes)

        )
        # Initialize weights
        self._init_weight()


    def forward(self, x):
        x = self.features(x)

        x = self.classifier(x)
        return x


    def _init_weight(self):
        # Initialize weights of original layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

@deprecated
def lenet_deep(**kwargs: Any) -> LeNet_Deep:
    model = LeNet_Deep(**kwargs)
    return model

class LeNet11_BN(nn.Module):
    def __init__(self, num_classes=10, wider_factor = 1, n_channels=1): 
        self.wider_factor = wider_factor
        print(f'Created a LeNet_Deep')
        super(LeNet11_BN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )


        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=wider_factor*512, out_features=256*wider_factor),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=256*wider_factor, out_features=128*wider_factor),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=128*wider_factor, out_features=num_classes)
        )
        # Initialize weights
        self._init_weight()


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


    def _init_weight(self):
        # Initialize weights of original layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

@deprecated
def lenet11_bn(**kwargs: Any) -> LeNet11_BN:
    model = LeNet11_BN(**kwargs)
    return model



class ResLeNet5_BN(nn.Module):
    def __init__(self, num_classes=10, wider_factor=8, n_channels=1):
        if wider_factor>1:
            print(f'Created a {wider_factor}Xwider ResLeNet5')
        super(ResLeNet5_BN, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=n_channels, out_channels=4*wider_factor, kernel_size=1, stride=1)

        self.conv1 = nn.Conv2d(in_channels=4*wider_factor, out_channels=16*wider_factor, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(16*wider_factor)
        self.relu2 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(in_channels=16*wider_factor, out_channels=16*wider_factor, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(16*wider_factor)
        self.downsample = nn.Sequential(
            nn.Conv2d(4*wider_factor, 16*wider_factor, kernel_size=5, stride=3, padding=0),
            nn.BatchNorm2d(16*wider_factor)
        )
        self.relu5 = nn.ReLU()
        self.pool6 = nn.MaxPool2d(2)



        self.fl7 = nn.Flatten()
        self.fc8 = nn.Linear(in_features=wider_factor*16 * 5 * 5, out_features=256*wider_factor)
        self.relu9 = nn.ReLU()
        self.fc10 = nn.Linear(in_features=256*wider_factor, out_features=128*wider_factor)
        self.relu11 = nn.ReLU()
        self.fc12 = nn.Linear(in_features=128*wider_factor, out_features=num_classes)

        # Initialize weights
        self._init_weight()


    def forward(self, x):
        xx = self.conv0(x)
        identity = xx
        out = self.conv1(xx)
        out = self.bn1(out)
        out = self.relu2(out)
        out = self.pool3(out)

        out = self.conv4(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(xx)
        out += identity
        out = self.relu5(out)
        out = self.pool6(out)

        out = self.fl7(out)

        out = self.fc8(out)
        out = self.relu9(out)
        out = self.fc10(out)
        out = self.relu11(out)

        out = self.fc12(out)

        return out


    def _init_weight(self):
        # Initialize weights of original layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


def reslenet5_bn(**kwargs: Any) -> ResLeNet5_BN:
    assert  'weights' in kwargs, "you should define model weight path"
    model = ResLeNet5_BN(**kwargs)
    model.load_state_dict(torch.load(kwargs['weights'], map_location='cpu'))
    print(f"Load reslenet5_bn weights from {kwargs['weights']}")
    model.consistent_reusable_permutation_map = {
        # reuse previous permutation instead of computing it for consistency reason. For exaxmple, in a residual block.
        "downsample.0":"conv4",
        # "conv4":"conv1"
    }
    return model