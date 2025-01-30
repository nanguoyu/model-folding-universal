from torchvision.models import VGG
from typing import Any
from functools import wraps
import torch
from torch import nn

from functools import partial
from typing import Any, cast, Dict, List, Optional, Union

__all__ = [
    'vgg11', 
    'vgg11_bn'
    ]

# For wider VGG11
def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# For wider VGG11
cfgs: Dict[str, List[Union[str, int]]] = {
    # 1x wider
    "1": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    # 2x wider
    "2": [64*2, "M", 128*2, "M", 256*2, 256*2, "M", 512*2, 512*2, "M", 512*2, 512, "M"],
    # 3x wider
    "3": [64*3, "M", 128*3, "M", 256*3, 256*3, "M", 512*3, 512*3, "M", 512*3, 512, "M"],
    # 4x wider
    "4": [64*4, "M", 128*4, "M", 256*4, 256*4, "M", 512*3, 512*4, "M", 512*4, 512, "M"],
}

# For wider VGG11
def _vgg(cfg: str, batch_norm: bool, weights, progress: bool, **kwargs: Any) -> VGG:
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model



def clean_vgg_kwargs(func):
    @wraps(func)
    def wrapper(**kwargs):
        valid_kwargs = ['features',  'num_classes', 'init_weights', 'dropout',
                        'weights', 'progress', 'wider_factor']
        extra_kwargs = {k: kwargs[k] for k in kwargs if k not in valid_kwargs}
        if extra_kwargs:
            print(f"Removed unsupported kwargs: {list(extra_kwargs.keys())}")
            kwargs = {k: kwargs[k] for k in kwargs if k in valid_kwargs}
        return func(**kwargs)
    return wrapper

@clean_vgg_kwargs
def vgg11(**kwargs: Any) -> VGG:
    from torchvision.models import  vgg11, VGG11_Weights

    assert 'num_classes' in kwargs, "you should define the num_classes"
    if kwargs['num_classes']!=1000:
        # Support the KEYWORDS of pretrained weights in Pytorch and our own model weight file path  with the same parameter name 'weights'
        pre_kwargs = {k: kwargs[k] for k in kwargs if k != 'weights'  and k!='wider_factor'}
        if kwargs['wider_factor'] ==1:
            model = vgg11(**pre_kwargs)
            print("Create vgg11: initialized")
        else:
            wider_factor = kwargs["wider_factor"]
            # _kwargs =  {k: pre_kwargs[k] for k in pre_kwargs if k != 'batch_norm'  and k!='progress'}
            model = _vgg(str(wider_factor), batch_norm=False, weights=None, progress=True, **pre_kwargs)
            print(f"Create vgg11_{wider_factor}XWider: initialized")

        if 'weights' in kwargs:
            model.load_state_dict(torch.load(kwargs['weights'], map_location='cpu'))
            print(f"Load vgg11 weights from {kwargs['weights']}")
    else:
        kwargs = {k: kwargs[k] for k in kwargs if k != 'wider_factor'}
        kwargs['weights']=VGG11_Weights.IMAGENET1K_V1
        print(f"Create vgg11: load pretrained weights {VGG11_Weights.IMAGENET1K_V1}.")
        model = vgg11(**kwargs)
    return model

@clean_vgg_kwargs
def vgg11_bn(**kwargs: Any) -> VGG:
    from torchvision.models import  vgg11_bn, VGG11_BN_Weights
    assert 'num_classes' in kwargs, "you should define the num_classes"
    if kwargs['num_classes']!=1000:
        # Support the KEYWORDS of pretrained weights in Pytorch and our own model weight file path  with the same parameter name 'weights'
        pre_kwargs = {k: kwargs[k] for k in kwargs if k != 'weights' and k != 'wider_factor'}
        # model = vgg11_bn(**pre_kwargs)
        # print("Create vgg11_bn: initialized")
        if kwargs['wider_factor'] ==1:
            model = vgg11_bn(**pre_kwargs)
            print("Create vgg11_bn: initialized")
        else:
            wider_factor = kwargs["wider_factor"]
            model = _vgg(str(wider_factor), batch_norm=True, weights=None, progress=True, **pre_kwargs)
            print(f"Create vgg11_bn_{wider_factor}XWider: initialized")

        if 'weights' in kwargs:
            model.load_state_dict(torch.load(kwargs['weights'], map_location='cpu'))
            print(f"Load vgg11_bn weights from {kwargs['weights']}")
    else:
        kwargs = {k: kwargs[k] for k in kwargs if k != 'wider_factor'}
        kwargs['weights']=VGG11_BN_Weights.IMAGENET1K_V1
        model = vgg11_bn(**kwargs)
        print(f"Create vgg11_BN: load pretrained weights {VGG11_BN_Weights.IMAGENET1K_V1}.")
    return model
