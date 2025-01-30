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
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding, LlamaAttention, LlamaMLP
from llama_2_7b_hf_recipe import llama_recipe as llama_recipe_2_7b
from llama_1_7b_hf_recipe import llama_recipe as llama_recipe_1_7b
from llama_3_2_1b_hf_recipe import llama_recipe as llama_recipe_3_2_1b
from utils import print_verbose, mix_weights_lerp, mix_weights
from core.weight_matching_utils import weight_distance_linear_conv2d, weight_distance_llama_attention, weight_distance_llama_mlp, load_model_graph
from core.permutation import permute_module

__all__ = ['weight_matching']


# --------- new version of weight matching ----------------- #

def weight_matching(model,device='cpu', distance_metric='l2', pairing_rate=1.0, nchannels=None, verbose=True, save_path=None, model_config_path=None, isllm=False, llm_recipe=None):
    """Find a permutation of `params_b` to make them match `params_a`."""
    print_verbose(verbose, 20*"-"+"perform weight matching"+20*"-")
    # This is an **expensive** operation for large model1 and model2 on GPUs.
    modela = copy.deepcopy(model).to(device)
    modelb = copy.deepcopy(model).to(device)
    
    model_graph, final_module_name = load_model_graph(model_config_path, nchannels, verbose=verbose)
    # todo: we may need to iterate permutation instead of module. In git-rebasin code, they randomly select a permuatation(related to multilayers) to compute the weight distance. See weight_m.py for more details.
    # But in our paper, we say we match layers from the first layer to the last layer.
    # todo: use L2-distance instead of scalar-distance. Mayeb define a distance function supporting L2-distance and scalar-distance.

    # todo: check if the final_module's num_channels in config is consistent with the last layer's num_channels in model.  !!!!!!!!!!!!!!
    params_a = modela.state_dict()
    params_b = modelb.state_dict()
    # print(params_a.keys())

    no_go_inside_the_module = []

    def not_go_insider(module_name):
        """
        For some modules such as LlamaAttention and LlamaMLP, we will go their submodules separately. 
        Because in LlamaAttention, we need to fuse qkv into one module and o as another module.
        In LlamaMLP, we need to fuse gate and up into one module and down into another module.
        """
        for name in no_go_inside_the_module:
            if name in module_name:
                return True
        return False


    # -------------create parameter_permutation_map------------------------------#
    print_verbose(verbose, 20*"-"+f"Perform weight matching and compute permutation"+20*"-")

    
    for module_name, module in modela.named_modules():
        if module_name == '' or not_go_insider(module_name) or isinstance(module, nn.Embedding) or module_name == final_module_name:
            continue
        if module_name not in model_graph:
            if not isinstance(module, (LlamaAttention, LlamaMLP)):
                # print(f"Skip {module_name}")
                continue
        
        # For LlamaAttention and LlamaMLP, we need to adjust the module_name
        module_name_ = module_name
        if isinstance(module, LlamaAttention):
            module_name_ = f"{module_name}_qkv"
        elif isinstance(module, LlamaMLP):
            module_name_ = f"{module_name}_gateup"
    

        if isllm:
            if module_name in llm_recipe:
                pr = llm_recipe[module_name]['pr']
            else:
                continue
        else:
            pr = pairing_rate
        
        # print(f"module_name: {module_name} {module_name_}")
        if is_batchnorm(module) or isinstance(module, LlamaRMSNorm): #todo: check if this is correct
            pre_name  = model_graph[module_name]['pre']
            model_graph[module_name]['permutation'] = model_graph[pre_name]['permutation']
            print_verbose(verbose, f'{module_name} reuses the same permutation from {pre_name}')
            continue

            # # ------------------------permute the module now-----------------------#
            # order = "output"
            # if isllm:
            #     assert llm_recipe is not None, "llm_recipe is not provided"
            #     if module_name in llm_recipe:
            #         permute_module(modelb, module, model_graph, module_name, order="output", verbose=verbose)
            #     else:
            #         print_verbose(verbose, f"Skip {module_name}")
            #         continue
            # else:
            #     permute_module(modelb, module, model_graph, module_name, order="output", verbose=verbose)
            # continue
        if is_batchnorm(module) or isinstance(module, LlamaRMSNorm) or is_pool(module) or isinstance(module, LlamaRotaryEmbedding) or len(list(module.named_parameters()))==0:
        #elif is_pool(module) or len(list(module.named_parameters()))==0:
            # There is no parameters to compute weight distance.
            # The following module should use the same permuatation of previous layer
            # 1. batchnorm. Note: batchnorm has weight, bias, running_mean, and running_var.
            # 2. pooling
            # 3. layernorm-like modules, such as layernorm, instancenorm, LlamaRotaryEmbedding, LlamaRMSNorm. Note: LlamaRotaryEmbedding has a buffer called inv_freq.
            # 4. other module without parameters, such as relu, sigmoid, silu, etc.
            pre_name  = model_graph[module_name]['pre']
            model_graph[module_name]['permutation'] = model_graph[pre_name]['permutation']
            print_verbose(verbose, f'{module_name} reuses the same permutation from {pre_name}. No folding on this layer.')
            continue

        if model_graph[module_name_]['consistent_map'] is not None:
            consistent_module_name = model_graph[module_name_]['consistent_map']
            assert model_graph[consistent_module_name]['permutation'] is not None, f"No permutation find in {consistent_module_name}. Please check the model_perm_config_path."
            model_graph[module_name_]['permutation'] = model_graph[consistent_module_name]['permutation']
            print_verbose(verbose, f"To matain consistency in a residual block, we apply {consistent_module_name}'s map on {module_name_}")
            # ------------------------permute the module now-----------------------#
            order = "output"
            if isllm:
                assert llm_recipe is not None, "llm_recipe is not provided"
                if module_name in llm_recipe:
                    permute_module(modelb, module, model_graph, module_name, order="output", verbose=verbose)
                else:
                    print_verbose(verbose, f"Skip {module_name}")
                    continue
            else:
                permute_module(modelb, module, model_graph, module_name, order="output", verbose=verbose)
            # ------------------------permute the module now-----------------------#
            # ------------------------average the weight---------------------------#
            # permuted_state_dict = modelb.state_dict()
            # mixed_sd = mix_weights_lerp(0.5,params_a, permuted_state_dict, device=device, method='weighted')
            # mixed_sd = mix_weights(params_a, permuted_state_dict, device=device)
            # modelb.load_state_dict(mixed_sd)
            continue

        if not model_graph[module_name_]['do_folding']:
            continue
        
        if save_path is not None:
            # pass
            permutation_module_path = f'{save_path}/{round(pr, 2)}/{module_name}_permutation_mtx.npy'
            if os.path.exists(permutation_module_path):
                # if you can find existing permutation, directly use it. Otherwise, we will compute it.
                existing_permutation = torch.from_numpy(np.load(permutation_module_path)).long()
                model_graph[module_name_]['permutation'] = existing_permutation
                print_verbose(verbose, f"Load existing permutation from {permutation_module_path}")
                # ------------------------permute the module now-----------------------#
                order = "output"
                print_verbose(verbose, f"Try to permute {module_name} {order}")  
                if isllm:
                    assert llm_recipe is not None, "llm_recipe is not provided"
                    if module_name in llm_recipe:
                        permute_module(modelb, module, model_graph, module_name, order="output", verbose=verbose)
                    else:
                        print_verbose(verbose, f"Skip {module_name}")
                        continue
                else:
                    permute_module(modelb, module, model_graph, module_name, order="output", verbose=verbose)
                continue
        #-------------------------For each module, compute the weight distance in the output channel.-----------------------------#
        print_verbose(verbose, f"Compute weight distance on {module_name_}'s output channel:")
        n = model_graph[module_name_]['num_channels']
        A = torch.zeros((n, n)).to(device)
        A.fill_diagonal_(-99999)

        if isinstance(module, LlamaAttention):
            A_this = weight_distance_llama_attention(module, module_name, model_graph, modela,  params_a, params_b, "output", n, device, distance_metric, verbose=verbose)
            no_go_inside_the_module.append(module_name) # not to go attention.q,k,v,o. We will go qkv and o separately.
        elif isinstance(module, LlamaMLP):
            no_go_inside_the_module.append(module_name) # not to go gate and up. We will go gate and up separately.
            A_this = weight_distance_llama_mlp(module, module_name, model_graph, modela,  params_a, params_b, "output", n, device, distance_metric, verbose=verbose)
        elif isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            A_this = weight_distance_linear_conv2d(module, module_name, model_graph,  modela, params_a, params_b, "output", n, device, distance_metric, verbose=verbose)
        else:
            raise ValueError(f"We don't match this type of module: {module_name}")
        A += A_this

        # -----------------------------Compute the weight distance on next modules' input channel-----------------------------#
        print_verbose(verbose, f"Compute weight distance on {module_name}'s next modules' input channel:")
        
        for next_module_name, value in model_graph.items():
            if value['pre'] == module_name:
                print_verbose(verbose, f"   Compute {module_name}'s next module {next_module_name}'s weight distance on input channel")
                next_module = dict(modela.named_modules())[next_module_name]
                if is_batchnorm(next_module) or is_pool(next_module) or isinstance(next_module, LlamaRMSNorm) or isinstance(next_module, LlamaRotaryEmbedding) or len(list(next_module.named_parameters()))==0:
                    continue
                elif isinstance(next_module, LlamaAttention):
                    A_next = weight_distance_llama_attention(next_module, next_module_name, model_graph, modela, params_a, params_b, "input", n, device, distance_metric, verbose=verbose)
                    A += A_next
                elif isinstance(next_module, LlamaMLP):
                    A_next = weight_distance_llama_mlp(next_module, next_module_name, model_graph, modela, params_a, params_b, "input", n, device, distance_metric, verbose=verbose)
                    A += A_next
                elif isinstance(next_module, nn.Linear) or isinstance(next_module, nn.Conv2d):
                    A_next = weight_distance_linear_conv2d(next_module, next_module_name, model_graph, modela, params_a, params_b, "input", n, device, distance_metric, verbose=verbose)
                    A += A_next
                else:
                    raise ValueError(f"We don't match this type of module: {next_module_name}")
                                        

        channel_pairs = greedy_channel_pairing(A, pairing_rate=pr)
        # print(f"channel_pairs: {channel_pairs}")
        # raise ValueError("stop here")

        ci = np.arange(len(A.detach().cpu().numpy()))
        for i, j in channel_pairs:
            ci[i] = j
            # ci[j] = i

        model_graph[module_name_]['permutation'] = torch.Tensor(ci).long()
        if  is_consecutive_increasing(ci):
            Warning(f"!!!!!!permutation is consecutive increasing for {module_name_}!!!!!!")

        # save permutation for cache
        if save_path is not None:
            os.makedirs(f'{save_path}/{round(pr, 2)}', exist_ok=True)
            permutation_module_path = f'{save_path}/{round(pr, 2)}/{module_name}_permutation_mtx.npy'
            print_verbose(verbose, f'Save permutation in {permutation_module_path}')
            np.save(permutation_module_path, torch.Tensor(ci).long().detach().cpu().numpy())

        # ------------------------permute the module now-----------------------#
        if  module_name_ != final_module_name:
            order = "output"
            if isllm:
                assert llm_recipe is not None, "llm_recipe is not provided"
                if module_name in llm_recipe:
                    permute_module(modelb, module, model_graph, module_name, order="output", verbose=verbose)
                else:
                    print_verbose(verbose, f"Skip {module_name}")
                    continue
            else:
                permute_module(modelb, module, model_graph, module_name, order="output", verbose=verbose)
            # ------------------------permute the module now-----------------------#
            # ------------------------average the weight---------------------------#
            # permuted_state_dict = modelb.state_dict()
            # mixed_sd = mix_weights_lerp(0.5,params_a, permuted_state_dict, device=device, method='weighted')
            # mixed_sd = mix_weights(params_a, permuted_state_dict, device=device)
            # modelb.load_state_dict(mixed_sd)


    # return model_graph, final_module_name # wrong
    # -----------------------------apply permutation-----------------------------#
    # permuted_state_dict = permute_model(modela=modela, modelb=modelb, model_graph=model_graph, final_module_name=final_module_name, loader=loader, device=device)
    # from matching.permutation import permute_model
    # permuted_state_dict = permute_model(modelb=modelb, model_graph=model_graph, final_module_name=final_module_name, device=device, isllm=isllm, llm_recipe=llm_recipe, verbose=verbose)

    del modela
    # del modelb
    torch.cuda.empty_cache()
    # return final_state_dict, model_graph, final_module_name
    return modelb, model_graph, final_module_name
