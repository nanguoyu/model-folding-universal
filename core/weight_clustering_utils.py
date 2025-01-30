from torch import nn
import torch
import numpy as np
from tqdm import tqdm
import copy
import json
import scipy
import os
from collections import OrderedDict

import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding, LlamaAttention, LlamaMLP
from llama_2_7b_hf_recipe import llama_recipe as llama_recipe_2_7b
from llama_1_7b_hf_recipe import llama_recipe as llama_recipe_1_7b
from llama_3_2_1b_hf_recipe import llama_recipe as llama_recipe_3_2_1b
from utils import print_verbose

from sklearn.cluster import AgglomerativeClustering, SpectralClustering, BisectingKMeans
from hkmeans import HKMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA



def load_model_graph(model_perm_config_path, verbose=True):
    """Load the model graph from the json file. Check: `do_folding`, `num_channels`, and `pre` for each module.
    """
    print_verbose(verbose, 20*"-"+f"Loading config map"+20*"-")
    with open(model_perm_config_path, 'r') as f:
        model_graph = json.load(f)
    final_module_name = model_graph.pop('final_module_name')
    
    # if nchannels is not None and model_graph['final_module_name']['num_channels'] != nchannels:
    #     model_graph['final_module_name']['num_channels'] = nchannels

    # init permutation map for all modules
    for module_name in model_graph.keys():
        num_channels = model_graph[module_name]['num_channels']
        if not model_graph[module_name]['do_folding']:
            if num_channels is None:
                # do not permute this module, there is no channel to permute
                continue
            else:
                # for embedding-like layers
                print(f"For {module_name}, we do not fold it. We set the merge to identity matrix.")
                model_graph[module_name]['merge'] = torch.eye(num_channels, dtype=torch.float16)
                continue

        model_graph[module_name]['merge'] = None #torch.eye(num_channels, dtype=torch.float16)
        # check if the module has a pre module.
        if module_name != 'input' and model_graph[module_name]['pre'] is None:
            raise ValueError(f"{module_name}'s pre is not defined. Please check the model config file: {model_perm_config_path}.")
        

    return model_graph, final_module_name

class WeightClustering:
    def __init__(self, n_clusters, n_features, device, normalize=False, use_kmeans=False):
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.normalize  = normalize
        self.use_kmeans = use_kmeans
        self.device = device
    def __call__(self, weight):
        km = AgglomerativeClustering(n_clusters=self.n_clusters)

        if self.use_kmeans is True:
            km = HKMeans(n_clusters=self.n_clusters, random_state=None, n_init=60,
                        n_jobs=-1, max_iter=10, verbose=True)

        X_scaled = weight.cpu().numpy()

        if self.normalize is True:
            scaler = preprocessing.RobustScaler().fit(X_scaled)
            X_scaled = scaler.transform(X_scaled)

        pca = PCA(n_components=weight.shape[0])
        X_scaled = pca.fit_transform(X_scaled)

        km.fit(X_scaled)

        matches = torch.zeros((self.n_features, self.n_clusters), device=self.device)

        for model_idx, match_idx in enumerate(km.labels_):
            matches[model_idx, match_idx] = 1
                                
        merge = matches.detach().clone() 

        m = merge.cpu().numpy()
        
        X_scaled = X_scaled / np.linalg.norm(X_scaled, axis=1)[:, None]
        inertia = (np.linalg.norm(
            X_scaled - m @ (np.diag(1.0 / np.diag(m.T @ m))) @ m.T @ X_scaled) ** 2) 

        return merge.T, inertia/X_scaled.shape[0]

# -----------------------------------Compute weight distance/merge------------------------------------------------------------#

def weight_distance_linear_conv2d(module: nn.Module, module_name, model_graph, model, params, order, n, device, verbose=True):
    """
    n: the number of channels in the original module.
    """
    if not isinstance(module, (nn.Linear, nn.Conv2d)):
        raise TypeError("module must be an instance of nn.Linear or nn.Conv2d")
    A = None
    if order == 'output':
        assert n == model_graph[module_name]['num_channels'], f"num_channels {n} != {model_graph[module_name]['num_channels']}" 
        for para_name, param in module.named_parameters():
            print_verbose(verbose, f'\t{module_name}.{para_name}')
            w = params[f'{module_name}.{para_name}']
            w = w.reshape((n, -1))
            if A is None:
                A = w
            else:
                A = torch.cat([A, w], dim=1)
        # Inlucde the weight of the next module which requires the current module's output
        print(f"compute {module_name}'s next module's input weight")
        for next_module_name, value in model_graph.items():
            if value['pre'] == module_name:
                next_module = dict(model.named_modules())[next_module_name]
                A_next = weight_distance_module(next_module, next_module_name, model_graph, model,params, order="input", n=n, device=device, verbose=verbose)
                if A_next is None:
                    print(f"No weight used for {next_module_name} {order}")
                    continue
                A = torch.cat([A, A_next], dim=1) if A_next is not None else A

    elif order == 'input':
        for para_name, param in module.named_parameters():
            w = params[f'{module_name}.{para_name}']
            if para_name == 'weight':
                print_verbose(verbose, f'\t{module_name}.{para_name}')
                # only for weight
                assert n == param.size()[1], f"num_channels {n} != {param.size()[1]}"
                w = torch.moveaxis(w, 1, 0).reshape((n, -1))
                if A is None:
                    A = w
                else:
                    A = torch.cat([A, w], dim=1)
    else:
        raise ValueError(f"Invalid order. Must be 'input' or 'output'.")
    return A

def weight_distance_llama_attention(module: LlamaAttention, module_name, model_graph, model, params, order, n, device, verbose=True):
    """
    n: the number of channels in the original module.
    n_folded: the number of channels you want to fold.
    """
    num_qkv_channels = model_graph[f"{module_name}_qkv"]["num_channels"]
    num_o_channels = model_graph[f"{module_name}_o"]["num_channels"]
    if module.num_key_value_groups==1:
        assert num_qkv_channels == module.num_heads, f"num_qkv_channels {num_qkv_channels} != {module.num_heads}"
    else:
        assert num_qkv_channels == module.num_key_value_heads, f"num_qkv_channels {num_qkv_channels} != {module.num_key_value_heads}"
    assert num_o_channels == module.hidden_size, f"num_o_channels {num_o_channels} != {module.hidden_size}"
    # in llama2 7b, num_qkv_channels = num_heads = hidden_size/head_dim = 4096/128 = 32
    # in llama2 7b, num_o_channels = hidden_size = 4096
    A = None
    if order == 'output':
        if module.num_key_value_groups==1:
            assert n == module.num_heads, f"num_heads {n} != {module.num_heads}"
        else:
            assert n == module.num_key_value_heads, f"num_key_value_heads {n} != {module.num_key_value_heads}"
        for sub_module_name in ['q_proj', 'k_proj', 'v_proj']:
            num_heads = module.num_heads if sub_module_name == 'q_proj' else module.num_key_value_heads
            if sub_module_name == 'q_proj' and module.num_key_value_groups>1:
                continue
            sub_module = dict(model.named_modules())[f'{module_name}.{sub_module_name}']
            for para_name, param in sub_module.named_parameters():
                print_verbose(verbose, f'\t{module_name}.{sub_module_name}.{para_name}')
                w = params[f'{module_name}.{sub_module_name}.{para_name}']
                # Weight shape of q is [num_heads * head_dim, hidden_size] in MHA and GQA
                # Weight shape of k, and v is [num_heads * head_dim, hidden_size] in MHA
                # Weight shape of k, and v is [num_kv_heads * head_dim, hidden_size] in GQA
                if para_name == 'weight':
                    # Firstly, reshape it to [num_heads, head_dim, hidden_size], then reshape it to [num_heads,-1]
                    w = w.reshape((num_heads, module.head_dim, module.hidden_size)).reshape((num_heads, -1))
                else :
                    # For bias
                    w = w.reshape((num_heads, module.head_dim, -1)).reshape((num_heads, -1))
                if A is None:
                    A = w
                else:
                    A = torch.cat([A, w], dim=1)
                
        for sub_module_name in ['o_proj']:
            # Weight shape of o is [num_heads * head_dim, hidden_size] in MHA and GQA
            if module.num_key_value_groups>1:
                continue
            sub_module = dict(model.named_modules())[f'{module_name}.{sub_module_name}']
            for para_name, param in sub_module.named_parameters():
                w = params[f'{module_name}.{sub_module_name}.{para_name}']
                # In case of permuting output channel of qkv, we aslo permute the input channel of o.
                if para_name == 'weight':
                    print_verbose(verbose, f'\t{module_name}.{sub_module_name}.{para_name}')
                    # only for input order according to q,k,v's permutation in output channel
                    # o's weight shape: (hidden_size, num_heads * head_dim)
                    w = torch.moveaxis(w, 1, 0).reshape((module.num_heads, module.head_dim, module.hidden_size)).reshape((module.num_heads, -1))
                    if A is None:
                        A = w
                    else:
                        A = torch.cat([A, w], dim=1)

        for next_module_name, value in model_graph.items():
            if value['pre'] == module_name:
                next_module = dict(model.named_modules())[next_module_name]
                A_next = weight_distance_module(next_module, next_module_name, model_graph, model,params, order="input", n=n, device=device, verbose=verbose)
                A = torch.cat([A, A_next], dim=1) if A_next is not None else A

    elif order == 'input':
        assert n == module.hidden_size, f"hidden_size {n} != {module.hidden_size}"
        for sub_module_name in ['q_proj', 'k_proj', 'v_proj']:
            sub_module = dict(model.named_modules())[f'{module_name}.{sub_module_name}']
            for para_name, param in sub_module.named_parameters():
                w = params[f'{module_name}.{sub_module_name}.{para_name}']
                if para_name == 'weight':
                    print_verbose(verbose, f'\t{module_name}.{sub_module_name}.{para_name}')
                    # we move the hidden_size to the first dim, and then reshape it to [hidden_size, -1]
                    w = torch.moveaxis(w, 1, 0).reshape((module.hidden_size, -1))
                    if A is None:
                        A = w
                    else:
                        A = torch.cat([A, w], dim=1)
    else:
        raise ValueError(f"Invalid order. Must be 'input' or 'output'.")
    return A

def weight_distance_llama_mlp(module: LlamaMLP, module_name, model_graph, model, params, order, n, device, verbose=True):
    """
    n: the number of channels in the original module.
    n_folded: the number of channels you want to fold.
    """
    num_gateup_channels = model_graph[f"{module_name}_gateup"]["num_channels"]
    num_down_channels = model_graph[f"{module_name}_down"]["num_channels"]
    assert num_gateup_channels == module.intermediate_size, f"num_gateup_channels {num_gateup_channels} != {module.intermediate_size}"
    assert num_down_channels == module.hidden_size, f"num_down_channels {num_down_channels} != {module.hidden_size}"
    # in llama2 7b, num_gateup_channels = intermediate_size = 11008
    # in llama2 7b, num_down_channels = hidden_size = 4096
    A = None
    if order == 'output':
        assert n == num_gateup_channels, f"num_gateup_channels {n} != {num_gateup_channels}"
        for sub_module_name in ['gate_proj','up_proj']:
            sub_module = dict(model.named_modules())[f'{module_name}.{sub_module_name}']
            for para_name, param in sub_module.named_parameters():
                print_verbose(verbose, f'\t{module_name}.{sub_module_name}.{para_name}')
                w = params[f'{module_name}.{sub_module_name}.{para_name}']
                # gate_proj and up_proj weight shape: (intermediate_size, hidden_size)
                w = w.reshape((num_gateup_channels, -1))
                if A is None:
                    A = w
                else:
                    A = torch.cat([A, w], dim=1)
        for sub_module_name in ['down_proj']:
            sub_module = dict(model.named_modules())[f'{module_name}.{sub_module_name}']
            for para_name, param in sub_module.named_parameters():
                w = params[f'{module_name}.{sub_module_name}.{para_name}']
                # In case of permuting output channel of gate and up, we aslo permute the input channel of down.
                if para_name == 'weight':
                    print_verbose(verbose, f'\t{module_name}.{sub_module_name}.{para_name}')
                    # down_proj weight shape: (hidden_size, intermediate_size)
                    # note: n == num_gateup_channels == intermediate_size
                    w = torch.moveaxis(w, 1, 0).reshape((num_gateup_channels, -1))
                    if A is None:
                        A = w
                    else:
                        A = torch.cat([A, w], dim=1)
        for next_module_name, value in model_graph.items():
            if value['pre'] == module_name:
                next_module = dict(model.named_modules())[next_module_name]
                A_next = weight_distance_module(next_module, next_module_name, model_graph, model,params, order="input", n=n, device=device, verbose=verbose)
                A = torch.cat([A, A_next], dim=1) if A_next is not None else A
    elif order == 'input':
        assert n == module.hidden_size, f"hidden_size {n} != {module.hidden_size}"
        for sub_module_name in ['gate_proj','up_proj']:
            sub_module = dict(model.named_modules())[f'{module_name}.{sub_module_name}']
            for para_name, param in sub_module.named_parameters():
                w = params[f'{module_name}.{sub_module_name}.{para_name}']
                if para_name == 'weight':
                    print_verbose(verbose, f'\t{module_name}.{sub_module_name}.{para_name}')
                    # gate_proj and up_proj weight shape: (intermediate_size, hidden_size)
                    # Note: note n==module.hidden_size
                    w = torch.moveaxis(w, 1, 0).reshape((module.hidden_size, -1))
                    if A is None:
                        A = w
                    else:
                        A = torch.cat([A, w], dim=1)
    else:
        raise ValueError(f"Invalid order. Must be 'input' or 'output'.") 
    return A

def weight_distance_module(module, module_name, model_graph, model, params, order, n, device, verbose=True):
    if isinstance(module, LlamaAttention):
        return weight_distance_llama_attention(module, module_name, model_graph, model, params, order, n, device, verbose=verbose)
    elif isinstance(module, LlamaMLP):
        return weight_distance_llama_mlp(module, module_name, model_graph, model, params, order, n, device, verbose=verbose)
    elif isinstance(module, (nn.Linear, nn.Conv2d)):
        return weight_distance_linear_conv2d(module, module_name, model_graph, model, params, order, n, device, verbose=verbose)
    else:
        print(f"Skip {module_name} {order}")
        return None


def get_weight_merge(module, module_name, model_graph, model, params, order, n, n_folded, device, normalize=False, use_kmeans=False,verbose=True):
    A = weight_distance_module(module, module_name, model_graph, model, params, order, n, device, verbose=verbose)
    print(f"A shape: {A.shape}")
    print(f"n_folded:{n_folded}, n:{n}")
    if A is None:
        return None, None
    # merge, inertia = weight_clustering(A, n_folded, n, device, normalize, use_kmeans)
    merger = WeightClustering(n_clusters=n_folded, n_features=A.shape[0], device=device, normalize=normalize, use_kmeans=use_kmeans)
    merge, inertia = merger(A)
    return merge, inertia

# -----------------------------------cluster the weight------------------------------------------------------------#
def cluster_llama_attention(model, model_graph, module_name, order: str, device,verbose=True):
    qkv_submodule_name = module_name + "_qkv"
    o_submodule_name = module_name + "_o"
    if not model_graph[qkv_submodule_name]['do_folding']:
        print_verbose(verbose, f"Skip {qkv_submodule_name}")
        return
    print_verbose(verbose, f"Try to cluster {module_name} {order}")
    with torch.no_grad():
        module = dict(model.named_modules())[module_name]
        assert isinstance(module, LlamaAttention), f"module {module_name} is not an instance of LlamaAttention"
        # in llama2 7b, num_qkv_channels = num_heads = hidden_size/head_dim = 4096/128 = 32
        # in llama2 7b, num_o_channels = hidden_size = 4096
        qkv_output_merge = model_graph[qkv_submodule_name]['merge'].to(torch.float16).to(device)
        print(o_submodule_name)
        o_output_merge = model_graph[o_submodule_name]['merge'].to(torch.float16).to(device)
        if order == 'output':
            for sub_module_name in ['q_proj', 'k_proj', 'v_proj']:
                num_heads = module.num_heads if sub_module_name == 'q_proj' else module.num_key_value_heads
                sub_module = dict(model.named_modules())[f'{module_name}.{sub_module_name}']
                if sub_module_name == 'q_proj' and module.num_key_value_groups>1:
                    # merge = scale_up_kv_merge(merge, module.num_key_value_groups)
                    # todo: implement it
                    raise NotImplementedError("Not implemented: scale_up_kv_merge(merge, module.num_key_value_groups)")
                for para_name, param in sub_module.named_parameters():
                    # Weight shape of q is [num_heads * head_dim, hidden_size] in MHA and GQA
                    # Weight shape of k, and v is [num_heads * head_dim, hidden_size] in MHA
                    # Weight shape of k, and v is [num_kv_heads * head_dim, hidden_size] in GQA
                    if para_name == 'weight':
                        reshaped_data = param.data.reshape(num_heads, module.head_dim, module.hidden_size)
                        shape = reshaped_data.shape
                        merged = (torch.diag(1.0 / torch.diag(qkv_output_merge @ qkv_output_merge.T))) @ qkv_output_merge @ reshaped_data.reshape(shape[0], -1)
                        merged = merged.reshape(qkv_output_merge.shape[0], *shape[1:])
                        param.data = merged.reshape(qkv_output_merge.shape[0]*module.head_dim, module.hidden_size)
                    else :
                        # For bias
                        param.data = (torch.diag(1.0 / torch.diag(qkv_output_merge @ qkv_output_merge.T))) @ qkv_output_merge @ param.data

                    print_verbose(verbose, f"\t{module_name}.{para_name} {order} Done.")
            sub_module_name = 'o_proj'
            o_module = dict(model.named_modules())[f'{module_name}.{sub_module_name}']
            # Weight shape of o is [hidden_size, num_heads * head_dim] in MHA and GQA
            if module.num_key_value_groups>1:
                # qkv_output_merge  = scale_up_kv_merge(qkv_output_merge, module.num_key_value_groups)
                raise NotImplementedError("Not implemented: scale_up_kv_merge(merge, module.num_key_value_groups)")
            for para_name, param in o_module.named_parameters():
                if para_name == 'weight':
                    # o's weight shape: (hidden_size, num_heads * head_dim)
                    # o's input
                    reshaped_data = param.data.permute(1, 0).reshape((module.num_heads, module.head_dim, module.hidden_size))
                    shape = reshaped_data.shape
                    merged = qkv_output_merge @ reshaped_data.reshape(shape[0], -1)
                    merged = merged.reshape(qkv_output_merge.shape[0], *shape[1:])
                    merged = merged.reshape(qkv_output_merge.shape[0]*module.head_dim, module.hidden_size)
                    param.data = merged.permute(1, 0)
                    # o's output
                    shape = param.data.shape
                    merged = (torch.diag(1.0 / torch.diag(o_output_merge @ o_output_merge.T))) @ o_output_merge @ param.data.reshape(shape[0], -1)
                    merged = merged.reshape(o_output_merge.shape[0], *shape[1:])
                    param.data = merged
                if para_name == 'bias':
                    # o's bias shape: (hidden_size)
                    # o's output
                    param.data = (torch.diag(1.0 / torch.diag(o_output_merge @ o_output_merge.T))) @ o_output_merge @ param.data
                print_verbose(verbose, f"\t{module_name}.{para_name} {order} Done.")
            # update num_heads and num_key_value_heads
            new_num_heads = qkv_output_merge.shape[0]
            module.num_heads = new_num_heads
            module.num_key_value_heads = new_num_heads # todo: check if this is correct
            print(f"merge {module_name}'s next module's input weight")
            for next_module_name, value in model_graph.items():
                if value['pre'] == module_name:
                    next_module = dict(model.named_modules())[next_module_name]
                    cluster_module(model, next_module, model_graph, next_module_name, order='input', device=device, verbose=verbose)
        elif order == 'input':
            for sub_module_name in ['q_proj', 'k_proj', 'v_proj']:
                num_heads = module.num_heads if sub_module_name == 'q_proj' else module.num_key_value_heads
                for param_name, param in module.named_parameters():
                    if param_name == 'weight':  
                        # qkv's input
                        pre_module_name = model_graph[qkv_submodule_name]['pre']
                        pre_merge = model_graph[pre_module_name]['merge']
                        pre_num_channels = model_graph[pre_module_name]['num_channels']
                        if param.size()[1] != pre_num_channels:
                            print_verbose(verbose, f"Find mismatch between {qkv_submodule_name}'s input channel number {param.size()[1]} and it's pre module {pre_module_name}'s output channel number {pre_num_channels}")
                            ratio  = int(param.size()[1] / pre_num_channels) 
                            new_merge = None
                            raise NotImplementedError("Not implemented: scale_up_merge")
                        else:
                            reshaped_data = param.data.permute(1,0)
                            shape = reshaped_data.shape
                            merged = pre_merge @ reshaped_data.reshape(shape[0], -1)
                            merged = merged.reshape(pre_merge.shape[0], *shape[1:])
                            param.data = merged.permute(1,0)
                    print_verbose(verbose, f"\t{module_name}.{param_name} {order} Done.") 
        else:
            raise ValueError(f"Invalid order. Must be 'input' or 'output'.")
                            
def cluster_llama_mlp(model, model_graph, module_name, order: str, device, verbose=True):
    #  name in config
    gateup_submodule_name = module_name + '_gateup'
    down_submodule_name = module_name + '_down'
    # name in model defination
    gate_layer_name = module_name+'.gate_proj'
    up_layer_name = module_name+'.up_proj'
    down_layer_name = module_name+'.down_proj'
    if not model_graph[gateup_submodule_name]['do_folding']:
        print_verbose(verbose, f"Skip {module_name}")
        return
    print_verbose(verbose, f"Try to cluster {module_name} {order}")
    with torch.no_grad():
        mlp_module = dict(model.named_modules())[module_name]
        hidden_size = mlp_module.hidden_size
        intermediate_size = mlp_module.intermediate_size
        gate_module = dict(model.named_modules())[gate_layer_name] # weight shape: [intermediate_size, hidden_size]
        up_module = dict(model.named_modules())[up_layer_name] # weight shape: [intermediate_size, hidden_size]
        down_module = dict(model.named_modules())[down_layer_name] # weight shape: [hidden_size, intermediate_size]

        gateup_merge = model_graph[gateup_submodule_name]['merge'].to(torch.float16).to(device)
        down_merge = model_graph[down_submodule_name]['merge'].to(torch.float16).to(device)
        if order == 'output':
            # merge the output of gate and up
            # for module in [gate_module, up_module]:
            for module_name in [gate_layer_name, up_layer_name]:
                module = dict(model.named_modules())[module_name]
                for param_name, param in module.named_parameters():
                    if param_name == 'weight':
                        original_shape = param.data.shape
                        merged = (torch.diag(1.0 / torch.diag(gateup_merge @ gateup_merge.T))) @ gateup_merge @ param.data.reshape(original_shape[0], -1)
                        merged = merged.reshape(gateup_merge.shape[0], *original_shape[1:])
                        param.data = merged
                    else:
                        param.data = (torch.diag(1.0 / torch.diag(gateup_merge @ gateup_merge.T))) @ gateup_merge @ param.data
                    print_verbose(verbose, f"\t{module_name}.{param_name} {order} Done.")
            # merge the input and output of down
            for param_name, param in down_module.named_parameters():
                if param_name == 'weight':
                    # down's input
                    reshaped_data = param.data.permute(1, 0)
                    shape = reshaped_data.shape
                    merged = gateup_merge @ reshaped_data.reshape(shape[0], -1)
                    merged = merged.reshape(gateup_merge.shape[0], *shape[1:])
                    param.data = merged.permute(1, 0)
                    # down's output
                    original_shape = param.data.shape
                    merged = (torch.diag(1.0 / torch.diag(down_merge @ down_merge.T))) @ down_merge @ param.data.reshape(original_shape[0], -1)
                    merged = merged.reshape(down_merge.shape[0], *original_shape[1:])
                    param.data = merged
                if param_name == 'bias':
                    # down's bias output
                    param.data = (torch.diag(1.0 / torch.diag(down_merge @ down_merge.T))) @ down_merge @ param.data
                print_verbose(verbose, f"\t{module_name}.{param_name} {order} Done.")
            print(f"merge {module_name}'s next module's input weight")
            for next_module_name, value in model_graph.items():
                if value['pre'] == module_name:
                    next_module = dict(model.named_modules())[next_module_name]
                    cluster_module(model, next_module, model_graph, next_module_name, order='input', device=device, verbose=verbose)
        elif order == 'input':
            for module in [gate_module, up_module]:
                for param_name, param in module.named_parameters():
                    if param_name == 'weight':
                        pre_module_name = model_graph[gateup_submodule_name]['pre']
                        pre_merge = model_graph[pre_module_name]['merge']
                        pre_num_channels = model_graph[pre_module_name]['num_channels']
                        if param.size()[1] != pre_num_channels:
                            print_verbose(verbose, f"Find mismatch between {module_name}'s input channel number {param.size()[1]} and it's pre module {pre_module_name}'s output channel number {pre_num_channels}")
                            raise NotImplementedError(f"Not implemented for pre-inconsistent channel number {order}") #todo: implement it
                        else:
                            reshaped_data = param.data.permute(1, 0)
                            shape = reshaped_data.shape
                            merged = pre_merge @ reshaped_data.reshape(shape[0], -1)
                            merged = merged.reshape(pre_merge.shape[0], *shape[1:])
                            param.data = merged.permute(1, 0)
                    print_verbose(verbose, f"\t{module_name}.{param_name} {order} Done.")
        else:
            raise ValueError(f"Invalid order. Must be 'input' or 'output'.")

def cluster_linear_conv2d(model, model_graph, module_name, order: str, device, verbose=True):
    if not model_graph[module_name]['do_folding']:
        print_verbose(verbose, f"Skip {module_name}")
        return
    print_verbose(verbose, f"Try to merge {module_name} {order}")
    with torch.no_grad():
        module = dict(model.named_modules())[module_name]
        if order == 'output':
            merge = model_graph[module_name]['merge'].to(torch.float32).to(device)
            for param_name, param in module.named_parameters():
                if param_name == 'weight':
                    original_shape = param.data.shape
                    merged = (torch.diag(1.0 / torch.diag(merge @ merge.T))) @ merge @ param.data.reshape(original_shape[0], -1)
                    param.data = merged.reshape(merge.shape[0], *original_shape[1:])
                    print(f"{original_shape} --> {param.data.shape}")
                if param_name == 'bias':
                    original_shape = param.data.shape
                    param.data = (torch.diag(1.0 / torch.diag(merge @ merge.T))) @ merge @ param.data
                    print(f"{original_shape} --> {param.data.shape}")
                print_verbose(verbose, f"\t{module_name}.{param_name} output Done.")
            print(f"merge {module_name}'s next module's input weight")
            for next_module_name, value in model_graph.items():
                if value['pre'] == module_name:
                    next_module = dict(model.named_modules())[next_module_name]
                    cluster_module(model, next_module, model_graph, next_module_name, order='input', device=device, verbose=verbose)
        elif order == 'input':
            pre_module_name = model_graph[module_name]['pre']
            pre_merge = model_graph[pre_module_name]['merge'].to(torch.float32).to(device)
            pre_num_channels = model_graph[pre_module_name]['num_channels']
            for param_name, param in module.named_parameters():
                if param_name == 'weight':
                    if param.size()[1] != pre_num_channels:
                        # In some cases after maxpooling, flattening, etc, the channel number is not consistent.
                        raise NotImplementedError(f"Not implemented for pre-inconsistent channel number {order}") #todo: implement it
                    original_shape = param.data.shape
                    if len(param.data.shape) == 2:
                        reshaped_data = param.data.permute(1, 0)
                    else:
                        reshaped_data = param.data.permute(1, 0, 2, 3)
                    shape = reshaped_data.shape
                    merged = pre_merge @ reshaped_data.reshape(shape[0], -1)
                    merged = merged.reshape(pre_merge.shape[0], *shape[1:])
                    if len(param.data.shape) == 2:
                        merged = merged.permute(1, 0)
                    else:
                        merged = merged.permute(1, 0, 2, 3)
                    param.data = merged
                    print(f"{original_shape} --> {param.data.shape}")
            print_verbose(verbose, f"\t{module_name}.{param_name} input Done.")
    
        else:
            raise ValueError(f"Invalid order. Must be 'input' or 'output'.")

def cluster_batchnorm2d(model, model_graph, module_name, order: str, device, verbose=True):
    if not model_graph[module_name]['do_folding']:
        print_verbose(verbose, f"Skip {module_name}")
        return
    print_verbose(verbose, f"Try to merge {module_name} {order}")
    with torch.no_grad():
        module = dict(model.named_modules())[module_name]
        if order == 'output':
            merge = model_graph[module_name]['merge'].to(torch.float32).to(device)

            for param_name, param in module.named_parameters():
                if param_name == 'weight':
                    original_shape = param.data.shape
                    merged = (torch.diag(1.0 / torch.diag(merge @ merge.T))) @ merge @ param.data.reshape(original_shape[0], -1)
                    merged = merged.reshape(merge.shape[0], *original_shape[1:])
                    param.data = merged
                if param_name == 'bias':
                    param.data = (torch.diag(1.0 / torch.diag(merge @ merge.T))) @ merge @ param.data
                print_verbose(verbose, f"\t{module_name}.{param_name} output Done.")
            running_mean = module.running_mean
            running_var = module.running_var
            
            inv_stds = 1.0/ torch.sqrt(running_var)
            new_inv_stds = (torch.diag(1.0 / torch.diag(merge @ merge.T))) @ merge @ inv_stds
            new_running_var = torch.square(1.0/(new_inv_stds))
             
            new_running_mean = running_mean * inv_stds
            new_running_mean  = (torch.diag(1.0 / torch.diag(merge @ merge.T))) @ merge @ new_running_mean
            new_running_mean = new_running_mean * torch.sqrt(new_running_var)

            module.running_mean = new_running_mean
            module.running_var = new_running_var
            print(f"merge {module_name}'s next module's input weight")
            for next_module_name, value in model_graph.items():
                if value['pre'] == module_name:
                    next_module = dict(model.named_modules())[next_module_name]
                    cluster_module(model, next_module, model_graph, next_module_name, order='input', device=device, verbose=verbose)
        elif order == 'input':
            print_verbose(verbose, f"Skip {module_name} {order}")
            return
        else:
            raise ValueError(f"Invalid order. Must be 'input' or 'output'.")

def cluster_llama_rms_norm(model, model_graph, module_name, order: str, device, verbose=True):
    if not model_graph[module_name]['do_folding']:
        print_verbose(verbose, f"Skip {module_name}")
        return
    print_verbose(verbose, f"Try to merge {module_name} {order}")
    with torch.no_grad():
        module = dict(model.named_modules())[module_name]
        if order == 'output':
            merge = model_graph[module_name]['merge'].to(torch.float16).to(device)

            for param_name, param in module.named_parameters():
                if param_name == 'weight':
                    original_shape = param.data.shape
                    merged = (torch.diag(1.0 / torch.diag(merge @ merge.T))) @ merge @ param.data.reshape(original_shape[0], -1)
                    merged = merged.reshape(merge.shape[0], *original_shape[1:])
                    param.data = merged
                if param_name == 'bias':
                    # LlamaRMSNorm has only one parameter: weight[hidden_size]
                    param.data = (torch.diag(1.0 / torch.diag(merge @ merge.T))) @ merge @ param.data
                print_verbose(verbose, f"\t{module_name}.{param_name} output Done.")
        elif order == 'input':
            print_verbose(verbose, f"Skip {module_name} {order}")
            return
        else:
            raise ValueError(f"Invalid order. Must be 'input' or 'output'.")

def cluster_llama_rotary_embedding(model, model_graph, module_name, order: str, device, verbose=True):
    if not model_graph[module_name]['do_folding']:
        print_verbose(verbose, f"Skip {module_name}")
        return
    # LlamaRotaryEmbedding has only one buffer: inv_freq[head_dim//2], 64
    # LlamaAttention's RotaryEmbedding will be removed v4.46 (RoPE is computed in the model, not in the decoder layers)
    # LlamaModel's rotary_emb is only applied after embedding layer, so we don't need to permute it.
    return

    

def cluster_module(model, module, model_graph, module_name, order: str, device, verbose=True):
    if isinstance(module, LlamaAttention):
        cluster_llama_attention(model, model_graph, module_name, order=order, device=device, verbose=verbose)
    elif isinstance(module, LlamaMLP):
        cluster_llama_mlp(model, model_graph, module_name, order=order, device=device, verbose=verbose)
    elif isinstance(module, (nn.Linear, nn.Conv2d)):
        cluster_linear_conv2d(model, model_graph, module_name, order=order, device=device, verbose=verbose)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        order = "output" # normally, we permute the output of batchnorm
        pre_name  = model_graph[module_name]['pre']
        model_graph[module_name]['merge'] = model_graph[pre_name]['merge']
        cluster_batchnorm2d(model, model_graph, module_name, order=order, device=device, verbose=verbose)
    elif isinstance(module, LlamaRMSNorm):
        order = "output" # normally, we permute the output of rmsnorm
        pre_name  = model_graph[module_name]['pre']
        model_graph[module_name]['merge'] = model_graph[pre_name]['merge']
        cluster_llama_rms_norm(model, model_graph, module_name, order=order, device=device, verbose=verbose)
    elif isinstance(module, LlamaRotaryEmbedding):
        cluster_llama_rotary_embedding(model, model_graph, module_name, order=order, device=device, verbose=verbose)
    else:
        # print(f"Skip {module_name} {order}")
        return