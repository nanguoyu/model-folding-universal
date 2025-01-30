import torch
import numpy as np
import torch.nn as nn
import torch.nn.utils.prune as prune
import os
import copy
import time
from tqdm import tqdm
import functools
import warnings

def deprecated(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(f"Function '{func.__name__}' is deprecated and will be removed in a future version.",
                      category=DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    return wrapper

def print_verbose(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)
    else:
        pass

def is_consecutive_increasing(arr):
  for i in range(len(arr) - 1):
    if arr[i+1] - arr[i] != 1:
      return False
  return True

def freeze(x):
    ret = copy.deepcopy(x)
    for key in x:
        ret[key] = x[key].detach()
    return ret

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def defreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True


def flatten_params(model):
    return model.state_dict()

def compare_models(model1, model2):
    """ 
    Check if two models are equal or not, including BatchNorm statistics
    """
    model1_sd = model1.state_dict()
    model2_sd = model2.state_dict()

    inconsistent_size_param_names = {}
    inconsistent_value_param_names = []
    inconsistent_bn_stats = {}

    isSameNames = True
    isSameValues = True
    isSameSizes = True
    isSameBNStats = True

    if model1_sd.keys() != model2_sd.keys():
        isSameNames = False

    keys1 = set(model1_sd.keys())
    keys2 = set(model2_sd.keys())
    intersection = keys1 & keys2

    for param_name in intersection:
        if 'num_batches_tracked' in param_name:
            continue
        # Check if parameter is from BatchNorm statistics
        is_bn_stat = 'running_mean' in param_name or 'running_var' in param_name
        
        if model1_sd[param_name].size() != model2_sd[param_name].size():
            inconsistent_size_param_names[param_name] = [model1_sd[param_name].size(), model2_sd[param_name].size()]
            isSameSizes = False
            print(f"find inconsistent size: {param_name} {model1_sd[param_name].size()} {model2_sd[param_name].size()}")

        if not torch.equal(model1_sd[param_name], model2_sd[param_name]):
            if is_bn_stat:
                inconsistent_bn_stats[param_name] = {
                    'model1': model1_sd[param_name],
                    'model2': model2_sd[param_name]
                }
                isSameBNStats = False
                # print(f"find inconsistent BN stat: {param_name}")
            else:
                inconsistent_value_param_names.append(param_name)
                isSameValues = False

    if isSameNames and isSameValues and isSameSizes and isSameBNStats:
        return True, "Equal models"
    else:
        msg = "\n[Log]"
        if not isSameNames:
            msg = msg+f"Different parameter names: {model1_sd.keys()} \n\n {model2_sd.keys()}\n"
        if not isSameSizes:
            msg = msg+f"The following parameters have different size:{inconsistent_size_param_names}\n\n"
        if not isSameValues:
            msg = msg+f"The following parameters have different values: {inconsistent_value_param_names}\n"
        if not isSameBNStats:
            msg = msg+f"The following BatchNorm statistics have different values: {list(inconsistent_bn_stats.keys())}\n"
        return False, msg

def sign_preserving_geometric_mean(w1, w2):
    sign = torch.sign(w1) * torch.sign(w2)
    return sign * torch.sqrt(torch.abs(w1 * w2))

def sign_preserving_harmonic_mean(w1, w2):
    sign = torch.sign(w1) * torch.sign(w2)
    return sign * (2 * torch.abs(w1) * torch.abs(w2)) / (torch.abs(w1) + torch.abs(w2))

def lerp(lam, t1, t2):
    import copy
    t3 = copy.deepcopy(t1)
    for p in t1:
        t3[p] = (1 - lam) * t1[p] + lam * t2[p]
    return t3


def mix_weights(sd0, sd1, device):
    sd_alpha = {k: sd0[k].to(device) + sd1[k].to(device)
                for k in sd0.keys()}
    return sd_alpha
def mix_weights_lerp(alpha, sd0, sd1, device, method='weighted'):
    if method == 'weighted':
        sd_alpha = {k: (1 - alpha) * sd0[k].to(device) + alpha * sd1[k].to(device)
                for k in sd0.keys()}
    elif method == 'geometric':
        sd_alpha = {
            k: sign_preserving_geometric_mean(sd0[k].to(device), sd1[k].to(device))
                for k in sd0.keys()}
    else:
        # default method: weighted
        sd_alpha = {k: (1 - alpha) * sd0[k].to(device) + alpha * sd1[k].to(device)
                for k in sd0.keys()}
        
    # model.load_state_dict(sd_alpha)
    return sd_alpha

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    # Ensure both tensors are at least 1D
    v0 = v0.flatten()
    v1 = v1.flatten()

    if not v0.is_floating_point():
        v0 = v0.to(torch.float32)
    if not v1.is_floating_point():
        v1 = v1.to(torch.float32)

    v0_norm = torch.linalg.norm(v0)
    v1_norm = torch.linalg.norm(v1)

    # Normalize vectors
    v0 = v0 / v0_norm if v0_norm > 0 else v0
    v1 = v1 / v1_norm if v1_norm > 0 else v1

    # Compute the cosine of the angle between the two vectors
    dot = torch.dot(v0, v1)

    # If the dot product is very close to 1, use linear interpolation
    if torch.abs(dot) > DOT_THRESHOLD:
        return (1 - t) * v0 + t * v1

    # Compute the angle between the vectors
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)

    # Compute the angle for the interpolation
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)

    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0

    return s0 * v0 + s1 * v1


def mix_weights_slerp(model, alpha, sd0, sd1, device):
    sd_alpha = {}
    for key in sd0.keys():
        v0, v1 = sd0[key].to(device), sd1[key].to(device)
        
        if v0.dim() == 2:  # FC
            merged = torch.stack([slerp(alpha, v0[i], v1[i]) for i in range(v0.size(0))])
        elif v0.dim() == 4:  # CONV
            merged = torch.stack([slerp(alpha, v0[i].flatten(), v1[i].flatten()).reshape_as(v0[i]) for i in range(v0.size(0))], dim=0)
        else:  # bias or others
            merged = slerp(alpha, v0, v1)
        
        sd_alpha[key] = merged
    
    # model.load_state_dict(sd_alpha)
    return sd_alpha

def local_structured_prune_model(model, pruning_rate=0.25, n=1, save_path=None):
    pruned_channels = {}
    total_num_parameters = 0
    total_num_pruned_parameters = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            total_num_parameters += module.weight.nelement()
            if n == 1:
                # L1范数
                importance = torch.sum(torch.abs(module.weight), dim=tuple(range(1, module.weight.dim())))
            elif n == 2:
                # L2范数
                importance = torch.sqrt(torch.sum(module.weight ** 2, dim=tuple(range(1, module.weight.dim()))))
            else:
                # 其他Ln范数
                importance = torch.sum(torch.abs(module.weight) ** n, dim=tuple(range(1, module.weight.dim()))) ** (1/n)
            if save_path is not None:
                os.makedirs(f'{save_path}/{pruning_rate}', exist_ok=True)
                importance_module_path = f'{save_path}/{pruning_rate}/{name}_importance.npy'
                print(f'Save importance in {importance_module_path}')
                np.save(importance_module_path, importance.detach().cpu().numpy())
        
            prune.ln_structured(module, name='weight', amount=pruning_rate, n=n, dim=0)
            # prune.ln_structured(module, name='bias', amount=pruning_rate, n=n, dim=0)
            prune.remove(module, 'weight')
            # prune.remove(module, 'bias')

            sparsity = 100. * float(torch.sum(module.weight == 0)) / float(module.weight.nelement())
            print(f"Sparsity in {name}.weight: {sparsity:.2f}%")
            total_num_pruned_parameters += torch.sum(module.weight == 0)
            
            if isinstance(module, nn.Conv2d):
                pruned_channels[name] = [i for i, w in enumerate(module.weight.detach().cpu().numpy()) if not w.any()]
            elif isinstance(module, nn.Linear):
                pruned_channels[name] = [i for i, w in enumerate(module.weight.detach().cpu().numpy()) if not w.any()]
            
            # print(f"Pruned channels in {name}: {pruned_channels[name]}")
            last_pruned_channels = [i for i, w in enumerate(module.weight.detach().cpu().numpy()) if not w.any()]

        elif isinstance(module, nn.BatchNorm2d):
            total_num_parameters += module.weight.nelement()
            if last_pruned_channels is not None:
                prune_mask = torch.ones(module.weight.data.shape).to(device=module.weight.data.device)
                prune_mask[last_pruned_channels] = 0
                module.weight.data.mul_(prune_mask)
                module.bias.data.mul_(prune_mask)
                module.running_mean.data.mul_(prune_mask)
                module.running_var.data.mul_(prune_mask)
                pruned_channels[name] = [i for i, w in enumerate(module.weight.detach().cpu().numpy()) if not w.any()]
                total_num_pruned_parameters += torch.sum(module.weight == 0)
            last_pruned_channels = None
    sparsity = 100. * float(total_num_pruned_parameters) / float(total_num_parameters)
    print(f"Total sparsity: {sparsity:.2f}%")
    return pruned_channels, sparsity

def local_unstructured_prune_model(model, pruning_rate=0.25):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=pruning_rate)
            prune.remove(module, 'weight')
            print(f"Sparsity in {name}.weight: {100. * float(torch.sum(module.weight == 0)) / float(module.weight.nelement()):.2f}%")

def replace_dropout(model):
    for i, layer in enumerate(model.classifier):
        if isinstance(layer, nn.Dropout):
            model.classifier[i] = nn.Identity()
    return model


def count_parameters(model, only_trainable=False):
    # redudant definition for compatibility.
    # TODO: remove it

    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def model_gpu_test(model, data_loader, device, num_epochs=3):
    """Test model inference speed and memory size on GPU
    
    Args:
        model: PyTorch model to test
        data_loader: DataLoader containing test data
        device: GPU device to use
        num_epochs: Number of epochs to test (default: 3)
        
    Returns:
        dict: Contains average GPU time and model size in MB
    """
    epoch_times = []
    
    # Calculate model size
    torch.cuda.empty_cache()
    model_size_before = torch.cuda.memory_allocated(device)
    model.to(device)
    model_size_after = torch.cuda.memory_allocated(device)
    model_mem = model_size_after - model_size_before
    
    model.eval()
    with torch.no_grad():
        for epoch in range(num_epochs+1):
            # Preload data to GPU
            batches = [(inputs.to(device), _) for inputs, _ in data_loader]
            
            # Time the inference
            torch.cuda.synchronize(device)
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            
            starter.record()
            for inputs, _ in tqdm(batches):
                outputs = model(inputs)
            ender.record()
            
            torch.cuda.synchronize(device)
            epoch_time = starter.elapsed_time(ender) / 1000.0  # Convert to seconds
            
            if epoch == 0:  # Skip first epoch (GPU warmup)
                continue
            epoch_times.append(epoch_time)
    
    avg_time = sum(epoch_times) / len(epoch_times)
    
    return {
        'avg_gpu_time': avg_time,
        'model_size': model_mem/1024**2  # Convert to MB
    }

def model_cpu_test(model, data_loader, num_epochs=3):
    """Test model inference speed on CPU
    
    Args:
        model: PyTorch model to test
        data_loader: DataLoader containing test data
        num_epochs: Number of epochs to test (default: 3)
        
    Returns:
        dict: Contains average CPU time
    """
    epoch_times = []
    model.to('cpu')
    model.eval()
    
    with torch.no_grad():
        for epoch in tqdm(range(num_epochs+1)):
            # Preload data to CPU
            batches = [(inputs.cpu(), _) for inputs, _ in data_loader]
            
            # Time the inference
            start_time = time.time()
            for inputs, _ in tqdm(batches):
                outputs = model(inputs)
            epoch_time = time.time() - start_time
            
            if epoch == 0:  # Skip first epoch (warmup)
                continue
            epoch_times.append(epoch_time)
    
    avg_time = sum(epoch_times) / len(epoch_times)
    
    return {
        'avg_cpu_time': avg_time
    }

def fold_to_prune_mask(permutation, num_channels):
    """
    Create a mask for channel pruning based on permutation matching.
    
    Args:
        permutation: Tensor containing channel mapping indices
        num_channels: Total number of channels
        
    Returns:
        Tensor: Boolean mask where True indicates channels to keep
    """
    mask = []
    visited = [False] * num_channels
    
    # Handle edge case
    if len(permutation) != num_channels:
        raise ValueError(f"Permutation length ({len(permutation)}) must match num_channels ({num_channels})")
        
    # Find channels to prune
    for i, j in enumerate(permutation):
        if not visited[i] and permutation[i] != i:
            if j >= num_channels:
                raise ValueError(f"Invalid permutation index {j} >= {num_channels}")
            mask.append(i)
            visited[i] = True
            visited[j] = True
            
    # Create the final mask
    pruning_mask = torch.zeros(num_channels, dtype=torch.bool)
    pruning_mask[mask] = True
    return ~pruning_mask  # Return channels to keep