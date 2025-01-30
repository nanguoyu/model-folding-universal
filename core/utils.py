from torch import nn
from torch.cuda.amp import GradScaler, autocast
import torch
import warnings
import functools
import numpy as np

import time
import functools

def timer_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.perf_counter()
        result = func(self, *args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        # class_name = self.__class__.__name__
        method_name = func.__name__
        print(f"{method_name} took {elapsed_time*1000:.3f} ms")
        # print(f"{class_name}.{method_name} took {elapsed_time*1000:.3f} ms")
        
        return result
    return wrapper

def find_asymmetric_elements(a, atol=1.3e-5, rtol=1e-8):
    if not isinstance(a, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Input must be a square matrix.")

    n = a.shape[0]
    asymmetric_elements = []

    for i in range(n):
        for j in range(i + 1, n):
            if np.abs(a[i, j] - a[j, i]) > atol + rtol * max(np.abs(a[i, j]), np.abs(a[j, i])):
                print(((i, j), (j, i), a[i, j], a[j, i]))
                asymmetric_elements.append(((i, j), (j, i), a[i, j], a[j, i]))

    return asymmetric_elements

def is_batchnorm(module):
    return isinstance(module, nn.modules.batchnorm._BatchNorm)

def is_layernorm(module):
    return isinstance(module, nn.modules.normalization._LayerNorm)

def is_maxpool(module):
    return isinstance(module, nn.modules.pooling._MaxPoolNd)
def is_avgpool(module):
    return isinstance(module, nn.modules.pooling._AvgPoolNd)
def is_pool(module):
    return is_maxpool(module) or is_avgpool(module)

def folding_layer_sparsity(permutation_map):
    num_paired_channels = 0
    num_channels = len(permutation_map)
    visited = [False] * num_channels
    
    for i, j in enumerate(permutation_map):
        if not visited[i] and not visited[j] and i != j and permutation_map[j] == i:
            num_paired_channels += 1
            visited[i] = True
            visited[j] = True
    
    return num_paired_channels, num_channels


def has_batchnorm2d(model):
    """
    Checks if a PyTorch model contains at least one BatchNorm2d layer

    Parameters:
    model (torch.nn.Module): The PyTorch model to check

    Returns:
    bool: True if the model contains at least one BatchNorm2d layer, False otherwise
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            return True
    return False

def has_batchnorm1d(model):
    """
    Checks if a PyTorch model contains at least one BatchNorm1d layer

    Parameters:
    model (torch.nn.Module): The PyTorch model to check

    Returns:
    bool: True if the model contains at least one BatchNorm1d layer, False otherwise
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            return True
    return False