from torch import nn
import torch
import numpy as np
from tqdm import tqdm
import copy
import json
import scipy
import os
from collections import OrderedDict
from core.pairing import greedy_channel_pairing
from core.utils import is_pool, is_batchnorm, is_avgpool, is_maxpool
from utils import is_consecutive_increasing
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding, LlamaAttention, LlamaMLP
from llama_2_7b_hf_recipe import llama_recipe as llama_recipe_2_7b
from llama_1_7b_hf_recipe import llama_recipe as llama_recipe_1_7b
from llama_3_2_1b_hf_recipe import llama_recipe as llama_recipe_3_2_1b
from utils import print_verbose

def load_model_graph(model_perm_config_path, nchannels=None, verbose=True):
    """Load the model graph from the json file. Check: 1.`do_folding`, `num_channels`, and `pre` for each module.
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
                model_graph[module_name]['permutation'] = torch.arange(num_channels)

        model_graph[module_name]['permutation'] = torch.arange(num_channels)
        # check if the module has a pre module.
        if module_name != 'input' and model_graph[module_name]['pre'] is None:
            raise ValueError(f"{module_name}'s pre is not defined. Algorithm fails.")
        

    return model_graph, final_module_name

def repeat_kv_weights(weight: torch.Tensor, n_rep: int, head_dim: int) -> torch.Tensor:
    """
    repeat the weight of k, and v in LlamaAttention.
    """
    num_kv_head_times_head_dim, hidden_size = weight.shape
    num_kv_heads = num_kv_head_times_head_dim // head_dim
    if n_rep == 1:
        return weight
    weight = weight.reshape(num_kv_heads, head_dim, hidden_size) 
    weight = weight[None,:,:,:].expand(n_rep, num_kv_heads, head_dim, hidden_size)
    weight = weight.reshape(num_kv_heads * n_rep, head_dim, hidden_size)
    return weight

def weight_distance(w_a, w_b, method='scalar'):
    if method == 'scalar':
        return w_a @ w_b.T
    elif method == 'l2':
        return -torch.cdist(w_a, w_b, p=2)
    elif method == 'cosine':
        return 1 - F.cosine_similarity(w_a, w_b, dim=1)
    else:
        raise ValueError(f"Invalid method. Must be 'scalar', 'l2', or 'cosine'.")

def weight_distance_linear_conv2d(module: nn.Module, module_name, model_graph, modela, params_a, params_b, order, n, device, distance_metric='l2',verbose=True):
    """
    n: the number of channels you want to match, could be the output channel or input channel.
    """
    if not isinstance(module, (nn.Linear, nn.Conv2d)):
        raise TypeError("module must be an instance of nn.Linear or nn.Conv2d")
    A = torch.zeros((n, n)).to(device)
    A.fill_diagonal_(-99999)
    if order == 'output':
        assert n == model_graph[module_name]['num_channels'], f"num_channels {n} != {model_graph[module_name]['num_channels']}" 
        for para_name, param in module.named_parameters():
            print_verbose(verbose, f'\t{module_name}.{para_name}')
            w_a = params_a[f'{module_name}.{para_name}']
            w_b = params_b[f'{module_name}.{para_name}']
            w_a = w_a.reshape((n, -1))
            w_b = w_b.reshape((n, -1))
            A+=weight_distance(w_a, w_b, method=distance_metric)
    elif order == 'input':
        for para_name, param in module.named_parameters():
            w_a = params_a[f'{module_name}.{para_name}']
            w_b = params_b[f'{module_name}.{para_name}']
            if para_name == 'weight':
                print_verbose(verbose, f'\t{module_name}.{para_name}')
                # only for weight
                assert n == param.size()[1], f"num_channels {n} != {param.size()[1]}"
                # if param.size()[1] != n:
                #     continue
                w_a = torch.moveaxis(w_a, 1, 0).reshape((n, -1))
                w_b = torch.moveaxis(w_b, 1, 0).reshape((n, -1))
                A+=weight_distance(w_a, w_b, method=distance_metric)
    else:
        raise ValueError(f"Invalid order. Must be 'input' or 'output'.")
    return A

def weight_distance_llama_attention(module: LlamaAttention, module_name, model_graph, modela, params_a, params_b, order, n, device, distance_metric='l2', verbose=True):
    """
    n: the number of channels you want to match, could be the output channel or input channel.
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
    A = torch.zeros((n, n)).to(device)
    A.fill_diagonal_(-99999)
    if order == 'output':
        if module.num_key_value_groups==1:
            assert n == module.num_heads, f"num_heads {n} != {module.num_heads}"
        else:
            assert n == module.num_key_value_heads, f"num_key_value_heads {n} != {module.num_key_value_heads}"
        for sub_module_name in ['q_proj', 'k_proj', 'v_proj']:
            num_heads = module.num_heads if sub_module_name == 'q_proj' else module.num_key_value_heads
            if sub_module_name == 'q_proj' and module.num_key_value_groups>1:
                continue
            sub_module = dict(modela.named_modules())[f'{module_name}.{sub_module_name}']
            for para_name, param in sub_module.named_parameters():
                print_verbose(verbose, f'\t{module_name}.{sub_module_name}.{para_name}')
                w_a = params_a[f'{module_name}.{sub_module_name}.{para_name}']
                w_b = params_b[f'{module_name}.{sub_module_name}.{para_name}']
                    # Weight shape of q is [num_heads * head_dim, hidden_size] in MHA and GQA
                    # Weight shape of k, and v is [num_heads * head_dim, hidden_size] in MHA
                    # Weight shape of k, and v is [num_kv_heads * head_dim, hidden_size] in GQA
                if para_name == 'weight':
                    # Firstly, reshape it to [num_heads, head_dim, hidden_size], then reshape it to [num_heads,-1]
                    w_a = w_a.reshape((num_heads, module.head_dim, module.hidden_size)).reshape((num_heads, -1))
                    w_b = w_b.reshape((num_heads, module.head_dim, module.hidden_size)).reshape((num_heads, -1))
                else :
                    # For bias
                    w_a = w_a.reshape((num_heads, module.head_dim, -1)).reshape((num_heads, -1))
                    w_b = w_b.reshape((num_heads, module.head_dim, -1)).reshape((num_heads, -1))
                A+=weight_distance(w_a, w_b, method=distance_metric)
                
        for sub_module_name in ['o_proj']:
            # Weight shape of o is [num_heads * head_dim, hidden_size] in MHA and GQA
            if module.num_key_value_groups>1:
                continue
            sub_module = dict(modela.named_modules())[f'{module_name}.{sub_module_name}']
            for para_name, param in sub_module.named_parameters():
                w_a = params_a[f'{module_name}.{sub_module_name}.{para_name}']
                w_b = params_b[f'{module_name}.{sub_module_name}.{para_name}']
                # In case of permuting output channel of qkv, we aslo permute the input channel of o.
                if para_name == 'weight':
                    print_verbose(verbose, f'\t{module_name}.{sub_module_name}.{para_name}')
                    # only for input order according to q,k,v's permutation in output channel
                    # o's weight shape: (hidden_size, num_heads * head_dim)
                    w_a = torch.moveaxis(w_a, 1, 0).reshape((module.num_heads, module.head_dim, module.hidden_size)).reshape((module.num_heads, -1))
                    w_b = torch.moveaxis(w_b, 1, 0).reshape((module.num_heads, module.head_dim, module.hidden_size)).reshape((module.num_heads, -1))
                    A+=weight_distance(w_a, w_b, method=distance_metric)

        # todo: should we also consider o_proj's next here or in weight_matching function?
        # todo: maybe not, as we only compute the distance of weight

    elif order == 'input':
        assert n == module.hidden_size, f"hidden_size {n} != {module.hidden_size}"
        for sub_module_name in ['q_proj', 'k_proj', 'v_proj']:
            sub_module = dict(modela.named_modules())[f'{module_name}.{sub_module_name}']
            for para_name, param in sub_module.named_parameters():
                w_a = params_a[f'{module_name}.{sub_module_name}.{para_name}']
                w_b = params_b[f'{module_name}.{sub_module_name}.{para_name}']
                if para_name == 'weight':
                    print_verbose(verbose, f'\t{module_name}.{sub_module_name}.{para_name}')
                    # we move the hidden_size to the first dim, and then reshape it to [hidden_size, -1]
                    w_a = torch.moveaxis(w_a, 1, 0).reshape((module.hidden_size, -1))
                    w_b = torch.moveaxis(w_b, 1, 0).reshape((module.hidden_size, -1))
                    A+=weight_distance(w_a, w_b, method=distance_metric)
        # nothing to do for o_proj
    else:
        raise ValueError(f"Invalid order. Must be 'input' or 'output'.")

    return A

def weight_distance_llama_mlp(module: LlamaMLP, module_name, model_graph, modela, params_a, params_b, order, n, device, distance_metric='l2', verbose=True):
    """
    n: the number of channels you want to match, could be the output channel or input channel.
    """
    num_gateup_channels = model_graph[f"{module_name}_gateup"]["num_channels"]
    num_down_channels = model_graph[f"{module_name}_down"]["num_channels"]
    assert num_gateup_channels == module.intermediate_size, f"num_gateup_channels {num_gateup_channels} != {module.intermediate_size}"
    assert num_down_channels == module.hidden_size, f"num_down_channels {num_down_channels} != {module.hidden_size}"
    # in llama2 7b, num_gateup_channels = intermediate_size = 11008
    # in llama2 7b, num_down_channels = hidden_size = 4096
    A = torch.zeros((n, n)).to(device)
    A.fill_diagonal_(-99999)
    if order == 'output':
        assert n == num_gateup_channels, f"num_gateup_channels {n} != {num_gateup_channels}"
        for sub_module_name in ['gate_proj','up_proj']:
            sub_module = dict(modela.named_modules())[f'{module_name}.{sub_module_name}']
            for para_name, param in sub_module.named_parameters():
                print_verbose(verbose, f'\t{module_name}.{sub_module_name}.{para_name}')
                w_a = params_a[f'{module_name}.{sub_module_name}.{para_name}']
                w_b = params_b[f'{module_name}.{sub_module_name}.{para_name}']
                # gate_proj and up_proj weight shape: (intermediate_size, hidden_size)
                w_a = w_a.reshape((num_gateup_channels, -1))
                w_b = w_b.reshape((num_gateup_channels, -1))
                A+=weight_distance(w_a, w_b, method=distance_metric)
        for sub_module_name in ['down_proj']:
            sub_module = dict(modela.named_modules())[f'{module_name}.{sub_module_name}']
            for para_name, param in sub_module.named_parameters():
                w_a = params_a[f'{module_name}.{sub_module_name}.{para_name}']
                w_b = params_b[f'{module_name}.{sub_module_name}.{para_name}']
                # In case of permuting output channel of gate and up, we aslo permute the input channel of down.
                if para_name == 'weight':
                    print_verbose(verbose, f'\t{module_name}.{sub_module_name}.{para_name}')
                    # down_proj weight shape: (hidden_size, intermediate_size)
                    # note: n == num_gateup_channels == intermediate_size
                    w_a = torch.moveaxis(w_a, 1, 0).reshape((num_gateup_channels, -1))
                    w_b = torch.moveaxis(w_b, 1, 0).reshape((num_gateup_channels, -1))
                    A+=weight_distance(w_a, w_b, method=distance_metric)
    elif order == 'input':
        assert n == module.hidden_size, f"hidden_size {n} != {module.hidden_size}"
        for sub_module_name in ['gate_proj','up_proj']:
            sub_module = dict(modela.named_modules())[f'{module_name}.{sub_module_name}']
            for para_name, param in sub_module.named_parameters():
                w_a = params_a[f'{module_name}.{sub_module_name}.{para_name}']
                w_b = params_b[f'{module_name}.{sub_module_name}.{para_name}']
                if para_name == 'weight':
                    print_verbose(verbose, f'\t{module_name}.{sub_module_name}.{para_name}')
                    # gate_proj and up_proj weight shape: (intermediate_size, hidden_size)
                    # Note: note n==module.hidden_size
                    w_a = torch.moveaxis(w_a, 1, 0).reshape((module.hidden_size, -1))
                    w_b = torch.moveaxis(w_b, 1, 0).reshape((module.hidden_size, -1))
                    A+=weight_distance(w_a, w_b, method=distance_metric)
    else:
        raise ValueError(f"Invalid order. Must be 'input' or 'output'.") 
    
    return A

