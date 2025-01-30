"""
Permutation operations for neural network model pruning.

This module implements various permutation functions to support model folding,
particularly for LLaMA-style transformer models. It handles permutation of weights
and parameters across different types of layers including attention, MLP, linear,
conv2d, batch normalization and RMS normalization layers.

Key functions:
- scale_up_kv_permutation: Scales up permutations for key/value heads in grouped attention
- permute_linear_conv2d: Handles permutation for linear and conv2d layers
- permute_llama_attention: Handles permutation for LLaMA attention layers
- permute_llama_mlp: Handles permutation for LLaMA MLP layers
- permute_llama_rms_norm: Handles permutation for LLaMA RMS normalization layers
- permute_model: Main entry point for permuting an entire model

The permutations are used to rearrange model parameters in a way that aligns the channels of the same layer.

Author: Dong Wang (dong.wang@tugraz.at)
Date: 2024-01-30
"""

import torch.nn as nn
import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding, LlamaAttention, LlamaMLP
import warnings
from utils import print_verbose, fold_to_prune_mask, is_consecutive_increasing

def scale_up_kv_permutation(kv_permutation, num_key_value_groups):
    """
    Scale up the permutation of k, and v in LlamaAttention.
    """
    # [i*num_key_value_groups + j for i in range(num_key_value_groups) for j  in kv_permutation]
    permutation = torch.tensor([i*len(kv_permutation) + j for i in range(num_key_value_groups) for j  in kv_permutation])

    # print(f"Permutation: {permutation}")
    # print(f"kv_permutation: {kv_permutation}")
    return permutation


def permute_linear_conv2d(modelb, model_graph, module_name, order: str, verbose=True):
    if not model_graph[module_name]['do_folding']:
        print_verbose(verbose, f"Skip {module_name}")
        return
    print_verbose(verbose, f"Try to permute {module_name} {order}")
    # num_paired_channels, num_channels = folding_layer_sparsity(model_graph[module_name]['permutation'])
    # print(f"Folding layer sparsity: {num_paired_channels} | {num_channels}")
    with torch.no_grad():
        module = dict(modelb.named_modules())[module_name]

        if order == 'output':
            for param_name, param in module.named_parameters():
                permutation = model_graph[module_name]['permutation']
                # permute the output of current module
                if model_graph[module_name]['consistent_map']:
                    reusable_perm_name = model_graph[module_name]['consistent_map']
                    permutation = model_graph[reusable_perm_name]['permutation']
                # param.copy_(param.index_select(0, permutation.to(param.device)))
                prune_mask = fold_to_prune_mask(permutation, model_graph[module_name]['num_channels'])
                merged_data = param.data + param.data.index_select(0, permutation.to(param.device))
                merged_data = merged_data/2
                indices = torch.where(prune_mask)[0]
                param.data = merged_data.index_select(0, indices.to(param.device))
                # print(f"indices: {indices}")
                print_verbose(verbose, f"\t{param_name} Done.")
            # permute the input of next modules
            print_verbose(verbose, f"\tfor {module_name}'s next modules...")
            for next_module_name, value in model_graph.items():
                if value['pre'] == module_name:
                    next_module = dict(modelb.named_modules())[next_module_name]
                    permute_module(modelb, next_module, model_graph, next_module_name, order='input',verbose=verbose)
        elif order == 'input':
            for param_name, param in module.named_parameters():
                permutation = model_graph[module_name]['permutation']
                if param_name == 'weight':
                    pre_module_name = model_graph[module_name]['pre']
                    pre_permutation = model_graph[pre_module_name]['permutation']
                    pre_num_channels = model_graph[pre_module_name]['num_channels']
                    # in case of flatten or pooling layer, the input channel number of next layer will be changed
                    if param.size()[1] != pre_num_channels:
                        print_verbose(verbose, f"Find mismatch between {module_name}'s input channel number {param.size()[1]} and it's pre module {pre_module_name}'s output channel number {pre_num_channels}")
                        ratio  = int(param.size()[1] / pre_num_channels) 
                        new_permutation = torch.tensor([i*ratio + j for i in pre_permutation  for j in range(ratio)], device=pre_permutation.device)
                        # param.copy_(param.index_select(1, new_permutation.to(param.device)))
                        prune_mask = fold_to_prune_mask(new_permutation, param.size()[1])
                        merged_data = param.data + param.data.index_select(1, new_permutation.to(param.device))
                        # merged_data = merged_data/2
                        # Convert boolean mask to indices using torch.where
                        indices = torch.where(prune_mask)[0]
                        param.data = merged_data.index_select(1, indices.to(param.device))
                    else:
                        # param.copy_(param.index_select(1, pre_permutation.to(param.device)))
                        prune_mask = fold_to_prune_mask(pre_permutation, pre_num_channels)
                        merged_data = param.data + param.data.index_select(1, pre_permutation.to(param.device))
                        # merged_data = merged_data/2
                        # Convert boolean mask to indices using torch.where
                        indices = torch.where(prune_mask)[0]
                        param.data = merged_data.index_select(1, indices.to(param.device))
                    print_verbose(verbose, f"\t{param_name} Done.")
        else:
            raise ValueError(f"Invalid order. Must be 'input' or 'output'.")


def permute_llama_attention(modelb, model_graph, module_name, order: str, verbose=True):
    # name in config
    qkv_submodule_name = module_name + '_qkv'
    o_submodule_name = module_name + '_o'
    if not model_graph[qkv_submodule_name]['do_folding']:
        print_verbose(verbose, f"Skip {module_name}")
        return
    print_verbose(verbose, f"Try to permute {module_name} {order}")
    # name in model defination
    q_layer_name = module_name+'.q_proj'
    k_layer_name = module_name+'.k_proj'
    v_layer_name = module_name+'.v_proj'
    o_layer_name = module_name+'.o_proj'
    # Weight shape of q is [num_heads * head_dim, hidden_size] in MHA and GQA
    # Weight shape of k, and v is [num_heads * head_dim, hidden_size] in MHA
    # Weight shape of k, and v is [num_kv_heads * head_dim, hidden_size] in GQA

    with torch.no_grad():
        attention_module = dict(modelb.named_modules())[module_name]
        assert isinstance(attention_module, LlamaAttention), f"Invalid module type: {type(attention_module)}"
        # num_heads = attention_module.num_heads
        head_dim = attention_module.head_dim
        hidden_size = attention_module.hidden_size  
        # If order == "outout"
        # first for q,k,v layer,we permute the their parameters according to model_graph[qkv_submodule_name]['permutation']
        # then we permute the input channel of o layer and also its output channel.
        # If order == "input"
        # we just permute the input of q,k,v layer.
        if order == 'output':
            # permute the output of q,k,v layer
            # for module in [q_module, k_module, v_module]:
            for sub_module_name in ['q_proj', 'k_proj', 'v_proj']:
                module = dict(modelb.named_modules())[f'{module_name}.{sub_module_name}']
                num_heads = attention_module.num_heads if sub_module_name == 'q_proj' else attention_module.num_key_value_heads
                permutation = model_graph[qkv_submodule_name]['permutation']

                if sub_module_name == 'q_proj' and attention_module.num_key_value_groups>1:
                    # print(f"Skip {module_name}.{sub_module_name} q_weight because num_key_value_groups > 1")
                    permutation = scale_up_kv_permutation(permutation, attention_module.num_key_value_groups)
          
                if model_graph[qkv_submodule_name]['consistent_map']:
                    reusable_perm_name = model_graph[qkv_submodule_name]['consistent_map']
                    permutation = model_graph[reusable_perm_name]['permutation']
                prune_mask = fold_to_prune_mask(permutation, num_heads)
                indices = torch.where(prune_mask)[0]
                num_pruned_heads = len(indices)
                for param_name, param in module.named_parameters():
                    # param.copy_(param.reshape(num_heads, head_dim, hidden_size).index_select(0, permutation.to(param.device)).reshape(num_heads * head_dim, hidden_size))
                    original_data = param.reshape(num_heads, head_dim, hidden_size).data
                    reshaped_data = original_data.index_select(0, permutation.to(param.device))
                    merged_data = (original_data + reshaped_data)/2
                    param.data = merged_data.index_select(0, indices.to(param.device)).reshape(num_pruned_heads * head_dim, hidden_size)
                    print_verbose(verbose, f"\t{sub_module_name}.{param_name} Done.")
                
            # permute o layer
            sub_module_name = 'o_proj'
            o_module = dict(modelb.named_modules())[f'{module_name}.{sub_module_name}']
            qkv_output_permutation = model_graph[qkv_submodule_name]['permutation']
            if attention_module.num_key_value_groups>1:
                qkv_output_permutation = scale_up_kv_permutation(qkv_output_permutation, attention_module.num_key_value_groups)
            prune_mask = fold_to_prune_mask(qkv_output_permutation, attention_module.num_heads)
            indices = torch.where(prune_mask)[0]
            num_pruned_heads = len(indices)
            for param_name, param in o_module.named_parameters():
                # first permute the input of o according to the output of q,k,v
                # weight shape: [hidden_size, num_heads * head_dim]
                # bias shape: [hidden_size]
                # param.copy_(param.reshape((hidden_size, attention_module.num_heads, head_dim)).index_select(1, qkv_output_permutation.to(param.device)).reshape((hidden_size, attention_module.num_heads * head_dim)))
                original_data = param.reshape((hidden_size, attention_module.num_heads, head_dim)).data
                reshaped_data = original_data.index_select(1, qkv_output_permutation.to(param.device))
                merged_data = (original_data + reshaped_data)/2
                # param.data = merged_data.reshape((hidden_size, attention_module.num_heads, head_dim)).index_select(1, indices.to(param.device)).reshape((hidden_size, num_pruned_heads * head_dim))
                param.data = merged_data.index_select(1, indices.to(param.device)).reshape((hidden_size, num_pruned_heads * head_dim))

                
                # then permute the output of o
                permutation = model_graph[o_submodule_name]['permutation']
                if model_graph[o_submodule_name]['consistent_map']:
                    reusable_perm_name = model_graph[o_submodule_name]['consistent_map']
                    permutation = model_graph[reusable_perm_name]['permutation']
                # param.copy_(param.index_select(0, permutation.to(param.device)))
                prune_mask = fold_to_prune_mask(permutation, model_graph[o_submodule_name]['num_channels']) #todo: check if this is correct
                merged_data = (param.data + param.data.index_select(0, permutation.to(param.device)))/2
                indices = torch.where(prune_mask)[0]
                param.data = merged_data.index_select(0, indices.to(param.device))

                print_verbose(verbose, f"\t{sub_module_name}.{param_name} Done.")
            attention_module.num_heads = num_pruned_heads
            attention_module.num_key_value_heads = num_pruned_heads # todo: check if this is correct
            # permute the input of next modules
            print_verbose(verbose, f"\tfor {o_submodule_name}'s next modules...")
            for next_module_name, value in model_graph.items():
                if value['pre'] == o_submodule_name:
                    print_verbose(verbose, f"\t\t {next_module_name}")  
                    next_module = dict(modelb.named_modules())[next_module_name]
                    permute_module(modelb, next_module, model_graph, next_module_name, order='input', verbose=verbose)
        elif order == 'input':
            # for module in [q_module, k_module, v_module]:
            for sub_module_name in ['q_proj', 'k_proj', 'v_proj']:
                module = dict(modelb.named_modules())[f'{module_name}.{sub_module_name}']
                num_heads = attention_module.num_heads if sub_module_name == 'q_proj' else attention_module.num_key_value_heads
                for param_name, param in module.named_parameters():
                    if param_name == 'weight':  
                        pre_module_name = model_graph[qkv_submodule_name]['pre']
                        pre_permutation = model_graph[pre_module_name]['permutation']
                        pre_num_channels = model_graph[pre_module_name]['num_channels']
                        if param.size()[1] != pre_num_channels:
                            print_verbose(verbose, f"Find mismatch between {qkv_submodule_name}'s input channel number {param.size()[1]} and it's pre module {pre_module_name}'s output channel number {pre_num_channels}")
                            ratio  = int(param.size()[1] / pre_num_channels) 
                            new_permutation = torch.tensor([i*ratio + j for i in pre_permutation  for j in range(ratio)], device=pre_permutation.device)
                            permutation = new_permutation
                            prune_mask = fold_to_prune_mask(permutation, param.size()[1])
                        else:
                            permutation = pre_permutation
                            prune_mask = fold_to_prune_mask(permutation, pre_num_channels)

                        assert param.shape == (num_heads * head_dim, hidden_size), f"Unexpected shape: {param.shape}"
                        # param.copy_(param.index_select(1,permutation.to(param.device)))
                        merged_data = param.data + param.data.index_select(1, permutation.to(param.device))
                        # merged_data = merged_data/2
                        indices = torch.where(prune_mask)[0]
                        param.data = merged_data.index_select(1, indices.to(param.device))
                        print_verbose(verbose, f"\t{param_name} Done.")
                    else:
                        continue
        else:
            raise ValueError(f"Invalid order. Must be 'input' or 'output'.")


def permute_llama_mlp(modelb, model_graph, module_name, order: str, verbose=True):
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
    print_verbose(verbose, f"Try to permute {module_name} {order}")
    with torch.no_grad():
        mlp_module = dict(modelb.named_modules())[module_name]
        hidden_size = mlp_module.hidden_size
        intermediate_size = mlp_module.intermediate_size
        gate_module = dict(modelb.named_modules())[gate_layer_name] # weight shape: [intermediate_size, hidden_size]
        up_module = dict(modelb.named_modules())[up_layer_name] # weight shape: [intermediate_size, hidden_size]
        down_module = dict(modelb.named_modules())[down_layer_name] # weight shape: [hidden_size, intermediate_size]

        permutation = model_graph[gateup_submodule_name]['permutation']
        if order == 'output':
            # permute the output of gate and up
            # for module in [gate_module, up_module]:
            for module_name in [gate_layer_name, up_layer_name]:
                module = dict(modelb.named_modules())[module_name]
                for param_name, param in module.named_parameters():
                    # weight shape: [intermediate_size, hidden_size]
                    # bias shape: [intermediate_size]
                    if model_graph[gateup_submodule_name]['consistent_map']: 
                        reusable_perm_name = model_graph[gateup_submodule_name]['consistent_map']
                        permutation = model_graph[reusable_perm_name]['permutation']
                    # param.copy_(param.index_select(0, permutation.to(param.device)))
                    prune_mask = fold_to_prune_mask(permutation, intermediate_size)
                    original_data = param.data
                    reshaped_data = original_data.index_select(0, permutation.to(param.device))
                    merged_data = (original_data + reshaped_data)/2
                    indices = torch.where(prune_mask)[0]
                    param.data = merged_data.index_select(0, indices.to(param.device))
                    print_verbose(verbose, f"\t {module_name}.{param_name} output Done.")
                    print_verbose(verbose, f"\t {original_data.shape} -> {param.data.shape}")
            # permute the input and output of down
            for param_name, param in down_module.named_parameters():
                # weight shape: [hidden_size, intermediate_size]
                # bias shape: [hidden_size]
                # first permute the input of down according to the output of gateup
                gateup_output_permutation = model_graph[gateup_submodule_name]['permutation']
                # param.copy_(param.index_select(1, gateup_output_permutation.to(param.device)))
                prune_mask = fold_to_prune_mask(gateup_output_permutation, intermediate_size)
                original_data = param.data
                reshaped_data = param.data.index_select(1, gateup_output_permutation.to(param.device))
                merged_data = (original_data+reshaped_data)/2
                indices = torch.where(prune_mask)[0]
                param.data = merged_data.index_select(1, indices.to(param.device))
                print_verbose(verbose, f"\t {down_layer_name}.{param_name} input Done.")
                print_verbose(verbose, f"\t {original_data.shape} -> {param.data.shape}")

                # then permute the output of down
                permutation = model_graph[down_submodule_name]['permutation']
                if model_graph[down_submodule_name]['consistent_map']:
                    reusable_perm_name = model_graph[down_submodule_name]['consistent_map']
                    permutation = model_graph[reusable_perm_name]['permutation']
                # param.copy_(param.index_select(0, permutation.to(param.device)))
                prune_mask = fold_to_prune_mask(permutation, hidden_size)
                original_data = param.data
                reshaped_data = param.data.index_select(0, permutation.to(param.device))
                merged_data = (original_data + reshaped_data)/2
                indices = torch.where(prune_mask)[0]
                param.data = merged_data.index_select(0, indices.to(param.device))
                print_verbose(verbose, f"\t{down_layer_name}.{param_name} output Done.")
                print_verbose(verbose, f"\t {original_data.shape} -> {param.data.shape}")

            # permute the input of next modules
            print_verbose(verbose, f"\tfor {down_submodule_name}'s next modules...")
            for next_module_name, value in model_graph.items():
                if value['pre'] == down_submodule_name:
                    print_verbose(verbose, f"\t\t {next_module_name}")  
                    next_module = dict(modelb.named_modules())[next_module_name]
                    permute_module(modelb, next_module, model_graph, next_module_name, order='input', verbose=verbose)
        elif order == 'input':
            for module in [gate_module, up_module]:
                for param_name, param in module.named_parameters():
                    if param_name == 'weight':
                        pre_module_name = model_graph[gateup_submodule_name]['pre']
                        pre_permutation = model_graph[pre_module_name]['permutation']
                        pre_num_channels = model_graph[pre_module_name]['num_channels']
                        if param.size()[1] != pre_num_channels:
                            print_verbose(verbose, f"Find mismatch between {gateup_submodule_name}'s input channel number {param.size()[1]} and it's pre module {pre_module_name}'s output channel number {pre_num_channels}")
                            ratio  = int(param.size()[1] / pre_num_channels) 
                            new_permutation = torch.tensor([i*ratio + j for i in pre_permutation  for j in range(ratio)], device=pre_permutation.device)
                            permutation = new_permutation
                            prune_mask = fold_to_prune_mask(permutation, param.size()[1])
                        else:
                            permutation = pre_permutation
                            prune_mask = fold_to_prune_mask(permutation, pre_num_channels)
                        assert param.shape == (intermediate_size, hidden_size), f"Unexpected shape: {param.shape}"
                        # param.copy_(param.index_select(1, permutation.to(param.device)))
                        merged_data = param.data + param.data.index_select(1, permutation.to(param.device))
                        # merged_data = merged_data/2
                        # Convert boolean mask to indices using torch.where
                        indices = torch.where(prune_mask)[0]
                        param.data = merged_data.index_select(1, indices.to(param.device))
                        print_verbose(verbose, f"\t{param_name} Done.")
        else:
            raise ValueError(f"Invalid order. Must be 'input' or 'output'.")
        
                    
def permute_llama_rms_norm(modelb, model_graph, module_name, order: str, verbose=True):
    if not model_graph[module_name]['do_folding']:
        print_verbose(verbose, f"Skip {module_name}")
        return
    print_verbose(verbose, f"Try to permute {module_name} {order}")
    # LlamaRMSNorm has only one parameter: weight[hidden_size]
    with torch.no_grad():
        rms_norm_module = dict(modelb.named_modules())[module_name]
        if order == 'output':
            permutation = model_graph[module_name]['permutation']
            if torch.equal(permutation, torch.arange(model_graph[module_name]['num_channels'], device=permutation.device)):
                print(f"same permutation,no need to fold {module_name}")
                return
            # print(f"len(permutation): {len(permutation)}")
            for param_name, param in rms_norm_module.named_parameters():
                # param.copy_(param.index_select(0, permutation.to(param.device)))
                prune_mask = fold_to_prune_mask(permutation, model_graph[module_name]['num_channels'])
                original_data = param.data
                merged_data = (original_data + original_data.index_select(0, permutation.to(param.device)))/2
                indices = torch.where(prune_mask)[0]
                # print(f"len(indices): {len(indices)}")
                param.data = merged_data.index_select(0, indices.to(param.device))
                print_verbose(verbose, f"\t{module_name}.{param_name} output Done.")
                print_verbose(verbose, f"\t {original_data.shape} -> {param.data.shape}")
        elif order == 'input':
            print_verbose(verbose, f"Skip {module_name} {order}")
            return
        else:
            raise ValueError(f"Invalid order. Must be 'input' or 'output'.")

def permute_llama_rotary_embedding(modelb, model_graph, module_name, order: str, verbose=True):
    if not model_graph[module_name]['do_folding']:
        print_verbose(verbose, f"Skip {module_name}")
        return
    # LlamaRotaryEmbedding has only one buffer: inv_freq[head_dim//2], 64
    # LlamaAttention's RotaryEmbedding will be removed v4.46 (RoPE is computed in the model, not in the decoder layers)
    # LlamaModel's rotary_emb is only applied after embedding layer, so we don't need to permute it.
    return

def permute_batchnorm2d(modelb, model_graph, module_name, order: str, verbose=True):
    if not model_graph[module_name]['do_folding']:
        print_verbose(verbose, f"Skip {module_name}")
        return
    print_verbose(verbose, f"Try to permute {module_name} {order}")
    with torch.no_grad():
        batchnorm_module = dict(modelb.named_modules())[module_name]
        device = batchnorm_module.running_mean.device
        if order == 'output':
            # permute the output of current module 
            permutation = model_graph[module_name]['permutation']
            if model_graph[module_name]['consistent_map']:
                reusable_perm_name = model_graph[module_name]['consistent_map']
                permutation = model_graph[reusable_perm_name]['permutation']
            prune_mask = fold_to_prune_mask(permutation, model_graph[module_name]['num_channels'])
            permutation = permutation.to(device)
            print(f"len(permutation): {len(permutation)}")
            for param_name, param in batchnorm_module.named_parameters():
                # print(f"bn param_name: {param_name}")
                # param.copy_(param.index_select(0, permutation.to(param.device)))
                print(f"len(param.data): {len(param.data)}")
                merged_data = param.data + param.data.index_select(0, permutation.to(param.device))
                # print(f"\tlen(prune_mask): {len(prune_mask)}")
                indices = torch.where(prune_mask)[0]
                # print(f"\tlen(indices): {len(indices)}")
                # print(f"indices: {indices}")
                param.data = merged_data.index_select(0, indices.to(param.device))
                print_verbose(verbose, f"\t{param_name} Done.")
            # print(f"permutation: {permutation}")

            # for running_mean and running_var
            # batchnorm_module.running_mean.copy_(batchnorm_module.running_mean[permutation])
            original_running_mean = batchnorm_module.running_mean
            merged_running_mean = (original_running_mean + original_running_mean.index_select(0, permutation.to(device)))/2
            batchnorm_module.running_mean = merged_running_mean.index_select(0, indices.to(device))
            # for i in range(len(merged_running_mean)):
            #     if merged_running_mean[i] != original_running_mean[i]:
            #         print(f"merged_running_mean[{i}] = {merged_running_mean[i]}, original_running_mean[{i}] = {original_running_mean[i]}")
    
            # batchnorm_module.running_var.copy_(batchnorm_module.running_var[permutation])
            original_running_var = batchnorm_module.running_var
            merged_running_var = (original_running_var + batchnorm_module.running_var.index_select(0, permutation.to(device)))/2
            batchnorm_module.running_var = merged_running_var.index_select(0, indices.to(device))
            # for i in range(len(merged_running_var)):
            #     if merged_running_var[i] != original_running_var[i]:
            #         print(f"merged_running_var[{i}] = {merged_running_var[i]}, original_running_var[{i}] = {original_running_var[i]}")

            print_verbose(verbose, f"\t{module_name}'s running_mean and running_var Done.")
            # permute the input of next modules
            # if module_name == "bn1":
            #     raise ValueError("Stop here")
            print_verbose(verbose, f"\tfor {module_name}'s next modules...")
            for next_module_name, value in model_graph.items():
                if value['pre'] == module_name:
                    print_verbose(verbose, f"\t\t {next_module_name}")  
                    next_module = dict(modelb.named_modules())[next_module_name]
                    permute_module(modelb, next_module, model_graph, next_module_name, order='input', verbose=verbose)
  
        elif order == 'input':
            print_verbose(verbose, f"Skip {module_name} {order}")
            return
        else:
            raise ValueError(f"Invalid order. Must be 'input' or 'output'.")




def permute_module(modelb, module, model_graph, module_name, order: str, verbose=True):
    if isinstance(module, LlamaAttention):
        permute_llama_attention(modelb, model_graph, module_name, order=order, verbose=verbose)
    elif isinstance(module, LlamaMLP):
        permute_llama_mlp(modelb, model_graph, module_name, order=order, verbose=verbose)
    elif isinstance(module, (nn.Linear, nn.Conv2d)):
        permute_linear_conv2d(modelb, model_graph, module_name, order=order, verbose=verbose)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        order = "output" # normally, we permute the output of batchnorm
        pre_name  = model_graph[module_name]['pre']
        model_graph[module_name]['permutation'] = model_graph[pre_name]['permutation']
        permute_batchnorm2d(modelb, model_graph, module_name, order=order, verbose=verbose)
    elif isinstance(module, LlamaRMSNorm):
        order = "output" # normally, we permute the output of rmsnorm
        pre_name  = model_graph[module_name]['pre']
        model_graph[module_name]['permutation'] = model_graph[pre_name]['permutation']
        permute_llama_rms_norm(modelb, model_graph, module_name, order=order, verbose=verbose)
    elif isinstance(module, LlamaRotaryEmbedding):
        permute_llama_rotary_embedding(modelb, model_graph, module_name, order=order, verbose=verbose)
    else:
        # print(f"Skip {module_name} {order}")
        return

def permute_model(modelb, model_graph, final_module_name, device, isllm, llm_recipe=None, verbose=True):
    print_verbose(verbose, "*"*10+"Permute model"+"*"*10)
    # permute modell  layer by layer
    for module_name, module in modelb.named_modules():
        if module_name not in model_graph:
            if not isinstance(module, (LlamaAttention, LlamaMLP)):
                print_verbose(verbose, f"Skip {module_name}")
                continue
        if len(list(module.named_parameters()))>0 and module_name != final_module_name:
            order = "output"
            if isllm:
                # for llama model
                assert llm_recipe is not None, "llm_recipe is not provided"
                if module_name in llm_recipe:
                    print_verbose(verbose, f"Try to permute {module_name} {order}") 
                    permute_module(modelb, module,model_graph, module_name, order="output", verbose=verbose)
                else:
                    print_verbose(verbose, f"Skip {module_name}")
                    continue
            else:
                permute_module(modelb, module,model_graph, module_name, order="output", verbose=verbose)
        else:
            continue
    return modelb.state_dict()
