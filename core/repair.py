"""
REPAIR (REctified Parameter Interpolation and Recalibration) Implementation
=======================================================================

This module implements REPAIR, a method for merging neural networks that interpolates
between model parameters while maintaining activation statistics. It includes both
the original REPAIR method and a closed-form variant.

The implementation is based on the paper:
"REPAIR: REnormalizing Permuted Activations for Interpolation Repair"

Key features:
- Tracks and adjusts activation statistics during model merging
- Supports both standard REPAIR and closed-form REPAIR
- Handles BatchNorm fusion and statistics recalibration

Original implementation forked from:
https://github.com/KellerJordan/REPAIR

Author: Dong Wang (dong.wang@tugraz.at)
Date: 2024-01-30
"""
from torch import nn
import torch
import copy
import numpy as np

from core.utils import is_batchnorm
#  forked from https://github.com/KellerJordan/REPAIR/blob/master/notebooks/Train-Merge-REPAIR-VGG11.ipynb


def reset_bn_stats(model, loader, device, epochs=1):
    num_data = 0
    # resetting stats to baseline first as below is necessary for stability
    for m in model.modules():
        if is_batchnorm(m):
            m.momentum = None # use simple average
            m.reset_running_stats()
    # run a single train epoch
    model.train()
    for _ in range(epochs):
        with torch.no_grad():
            for images, _ in loader:
                output = model(images.to(device))
                num_data+=len(images)
                if num_data>=1000:
                    print("Enough data for REPAIR")
                    break
    model.eval()
    return model


def model_keys(model):
    sd = model.state_dict()
    return sd.keys()

class TrackLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.bn = nn.BatchNorm2d(layer.out_channels)
        
    def get_stats(self):
        return (self.bn.running_mean, self.bn.running_var.sqrt())
        
    def forward(self, x):
        x1 = self.layer(x)
        self.bn(x1)
        return x1

class ResetLayer(nn.Module):
    def __init__(self, layer, name=None):
        super().__init__()
        self.name = name
        self.layer = layer
        self.bn = nn.BatchNorm2d(layer.out_channels)
        
    def set_stats(self, goal_mean, goal_std):
        self.bn.bias.data = goal_mean
        self.bn.weight.data = goal_std
        
    def forward(self, x):
        x1 = self.layer(x)
        return self.bn(x1)

# adds TrackLayers around every conv layer
def make_tracked_net(net, device, pre_name):
    """
    Recursively adds a ResetLayer after all Conv2d layers in net.
    This function handles modules without children by directly modifying them if needed.
    """
    for name, module in list(net.named_children()):  # Use list to avoid modification issues during iteration
        # Directly modify the module if it is a Conv2d (handling modules without children)
        if isinstance(module, nn.Conv2d):
            print(f"attach a tracker wrapper to {pre_name}.{name}")
            reset_layer = TrackLayer(module).to(device)
            setattr(net, name, reset_layer)  # Replace the Conv2d module with ResetLayer in the parent module
            # print(net)
        else:
            # Recursively call this function on children modules
            make_tracked_net(module, device, pre_name=f"{pre_name}.{name}")
    net.to(device)
    return net.eval()


# adds ResetLayers around every conv layer
def make_repaired_net(net, device, pre_name):
    """
    In-placely, add resetlayer after all conv layers in net1.
    """
    for name, module in list(net.named_children()):  # Use list to avoid modification issues during iteration
        # Directly modify the module if it is a Conv2d (handling modules without children)
        if isinstance(module, nn.Conv2d):
            print(f"attach a reset wrapper to {pre_name}.{name}")
            reset_layer = ResetLayer(module, name=f'{pre_name}.{name}').to(device)
            setattr(net, name, reset_layer)  # Replace the Conv2d module with ResetLayer in the parent module
            # print(net)
        else:
            # Recursively call this function on children modules
            make_repaired_net(module, device, pre_name=f"{pre_name}.{name}")
    return net.eval()

def fuse_conv_bn(conv, bn):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 bias=True if conv.bias is not None else False) 

    # set weights
    w_conv = conv.weight.clone()
    bn_std = (bn.eps + bn.running_var).sqrt()
    gamma = bn.weight / bn_std
    fused_conv.weight.data = (w_conv * gamma.reshape(-1, 1, 1, 1))

    # set bias
    if conv.bias is not None:
        beta = bn.bias + gamma * (-bn.running_mean + conv.bias)
        fused_conv.bias.data = beta
    
    return fused_conv


def fuse_tracked_net(net, net1, device, pre_name=''):
    # net1 is an original model 
    # net is a reset model wrapper
    for name, module in net.named_children():
        full_name = f"{pre_name}.{name}" if pre_name else name
        # print(f"Process {full_name}")
        # print(module)
        # locate module in net1
        target_module = net1
        for sub_name in full_name.split('.'):
            if hasattr(target_module, sub_name):
                target_module = getattr(target_module, sub_name)
            else:
                print(f"Module {full_name} not found in net1.")
                return net1  # or, continue to others

        if isinstance(module, ResetLayer):
            print(f"fuse ResetLayer module {full_name} to conv")
            fused_conv = fuse_conv_bn(module.layer, module.bn).to(device)
            # determine target_module's parents module and current module's name
            parent_name, child_name = full_name.rsplit('.', 1) if '.' in full_name else ('', full_name)
            parent_module = net1
            if parent_name:  
                for sub_name in parent_name.split('.'):
                    parent_module = getattr(parent_module, sub_name)
            setattr(parent_module, child_name, fused_conv)
        else:
            # recurssivly process children modules.
            # print(f"go to {full_name}")
            fuse_tracked_net(module, net1, device, pre_name=full_name)
    return net1

def repair(modela, modelb, model_mix, loader, device, alpha=0.5):
    # Note: model0 and model1 will be modified in-place.
    model0=copy.deepcopy(modela)
    model1=copy.deepcopy(modelb)
    # model_mix will be returned
    
    model_mix_copy = copy.deepcopy(model_mix)
    print(f"REPAIR starting...")
    ## Calculate all neuronal statistics in the endpoint networks
    # print(f"model0:\n{model0}")
    # print(model_keys(model0))
    wrap0 = make_tracked_net(model0, device, pre_name="modela")
    # print(f"wrap0:\n{wrap0}")
    # print(model_keys(wrap0))
    # print("----")
    # print(f"model1:\n{model1}")
    # print(model_keys(model1))
    wrap1 = make_tracked_net(model1, device, pre_name="modelb")
    # print(f"wrap1:\n{wrap1}")
    # print(model_keys(wrap1))

    reset_bn_stats(wrap0, loader, device)
    reset_bn_stats(wrap1, loader, device)
    
    # print(f"model_mix:\n{model_mix}")
    # print(model_keys(model_mix))
    wrap_a = make_repaired_net(model_mix_copy, device, pre_name="modelmix")
    # print(f"wrap_a:\n{wrap_a}")
    # print(model_keys(wrap_a))

    # Iterate through corresponding triples of (TrackLayer, TrackLayer, ResetLayer)
    # around conv layers in (model0, model1, model_a).
    for track0, track1, reset_a in zip(wrap0.modules(), wrap1.modules(), wrap_a.modules()): 
        if not isinstance(track0, TrackLayer):
            continue
        assert (isinstance(track0, TrackLayer)
                and isinstance(track1, TrackLayer)
                and isinstance(reset_a, ResetLayer))
        print(reset_a.name[len("modelmix."):])
        print(f"reset the net's stats")
        # print(track0, track1, reset_a)# 
        # print("----")
        # get neuronal statistics of original networks
        mu0, std0 = track0.get_stats()
        mu1, std1 = track1.get_stats()
        # set the goal neuronal statistics for the merged network 
        goal_mean = (1 - alpha) * mu0 + alpha * mu1
        goal_std = (1 - alpha) * std0 + alpha * std1
        reset_a.set_stats(goal_mean, goal_std)

    # print(f"wrap_a after :\n{wrap_a}")
    # print(model_keys(wrap_a))
    # Estimate mean/vars such that when added BNs are set to eval mode,
    # neuronal stats will be goal_mean and goal_std.
    reset_bn_stats(wrap_a, loader, device)
    # fuse the rescaling+shift coefficients back into conv layers
    # print(f"model_mix: \n {model_mix}")
    # print(model_mix.features[0].bias)
    model_b = fuse_tracked_net(wrap_a, model_mix, device, pre_name="")
    # print(f"model_mix after fusion: \n {model_mix}")
    # print(model_mix.features[0].bias)
    # print(f"model_b:\n {model_b}")
    # print(model_keys(model_b))
    
    return model_b


def repair_closedform(modela, modelb, model_mix, loader, device, corr_path, alpha=0.5):
    # Forked from https://github.com/KellerJordan/REPAIR/blob/master/notebooks/Train-Merge-REPAIR-VGG11-ClosedForm.ipynb by Keller Jordan
    # Note: model0 and model1 will be modified in-place.
    model0=copy.deepcopy(modela)
    model1=copy.deepcopy(modelb)
    # model_mix will be returned
    
    model_mix_copy = copy.deepcopy(model_mix)
    print(f"REPAIR starting...")
    ## Calculate all neuronal statistics in the endpoint networks
    # print(f"model0:\n{model0}")
    # print(model_keys(model0))
    wrap0 = make_tracked_net(model0, device, pre_name="modela")
    # print(f"wrap0:\n{wrap0}")
    # print(model_keys(wrap0))
    # print("----")
    # print(f"model1:\n{model1}")
    # print(model_keys(model1))
    wrap1 = make_tracked_net(model1, device, pre_name="modelb")
    # print(f"wrap1:\n{wrap1}")
    # print(model_keys(wrap1))

    reset_bn_stats(wrap0, loader, device)
    reset_bn_stats(wrap1, loader, device)

    # 
    # wrap_a = make_repaired_net(model_mix_copy, device, pre_name="modelmix")
    # for track0, track1, reset_a in zip(wrap0.modules(), wrap1.modules(), wrap_a.modules()): 
    #     if not isinstance(track0, TrackLayer):
    #         continue
    #     assert (isinstance(track0, TrackLayer)
    #             and isinstance(track1, TrackLayer)
    #             and isinstance(reset_a, ResetLayer))
    #     print(f"reset the net's stats")
    #     # print(track0, track1, reset_a)# 
    #     # print("----")
    #     # get neuronal statistics of original networks
    #     mu0, std0 = track0.get_stats()
    #     mu1, std1 = track1.get_stats()
    #     # set the goal neuronal statistics for the merged network 
    #     goal_mean = (1 - alpha) * mu0 + alpha * mu1
    #     goal_std = (1 - alpha) * std0 + alpha * std1
    #     reset_a.set_stats(goal_mean, goal_std)

    # 
    wrap_a = make_repaired_net(model_mix_copy, device, pre_name="modelmix")
    # Iterate through corresponding triples of (TrackLayer, TrackLayer, ResetLayer)
    # around conv layers in (model0, model1, model_a).
    # corr_vectors = None
    # corr_vec_it = iter(corr_vectors)
    for track0, track1, reset_a in zip(wrap0.modules(), wrap1.modules(), wrap_a.modules()): 
        if not isinstance(track0, TrackLayer):
            continue  
        assert (isinstance(track0, TrackLayer)
                and isinstance(track1, TrackLayer)
                and isinstance(reset_a, ResetLayer))

        print(reset_a.name[len("modelmix."):])
        k = reset_a.name[len("modelmix."):]
        corr_vec = np.load( f'{corr_path}/{k}_corr_vector.npy')
        corr_vec = torch.tensor(corr_vec).to(device)
        # todo: load  corr_vec from somthing according to name
        # get neuronal statistics of original networks
        mu0, std0 = track0.get_stats()
        mu1, std1 = track1.get_stats()
        # set the goal neuronal statistics for the merged network 
        goal_mean = (1 - alpha) * mu0 + alpha * mu1
        goal_std = (1 - alpha) * std0 + alpha * std1
        
        exp_mean = goal_mean
        exp_std = ((1-alpha)**2*std0**2 + alpha**2*std1**2 + 2*alpha*(1-alpha)*std0*std1*corr_vec)**0.5
        goal_std_ratio = goal_std / exp_std
        goal_mean_shift = goal_mean - goal_std_ratio * exp_mean
        
        # Y = aX + b, where X has mean/var mu/sigma^2, and we want nu/tau^2,
        # so we set a = tau/sigma and b = nu - (tau / sigma) mu
        
        reset_a.set_stats(goal_mean_shift, goal_std_ratio)

    # 

    # print(f"wrap_a after :\n{wrap_a}")
    # print(model_keys(wrap_a))
    # Estimate mean/vars such that when added BNs are set to eval mode,
    # neuronal stats will be goal_mean and goal_std.
    reset_bn_stats(wrap_a, loader, device)
    # fuse the rescaling+shift coefficients back into conv layers
    # print(f"model_mix: \n {model_mix}")
    # print(model_mix.features[0].bias)
    model_b = fuse_tracked_net(wrap_a, model_mix, device, pre_name="")
    # print(f"model_mix after fusion: \n {model_mix}")
    # print(model_mix.features[0].bias)
    # print(f"model_b:\n {model_b}")
    # print(model_keys(model_b))
    
    return model_b