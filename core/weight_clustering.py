from torch import nn
import torch
import numpy as np
from tqdm import tqdm
import copy
import json
import scipy
import os
from collections import OrderedDict
from core.utils import is_pool, is_batchnorm, is_avgpool, is_maxpool
from utils import is_consecutive_increasing
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding, LlamaAttention, LlamaMLP
from llama_2_7b_hf_recipe import llama_recipe as llama_recipe_2_7b
from llama_1_7b_hf_recipe import llama_recipe as llama_recipe_1_7b
from llama_3_2_1b_hf_recipe import llama_recipe as llama_recipe_3_2_1b
from utils import print_verbose, mix_weights_lerp, mix_weights
from core.weight_clustering_utils import load_model_graph, get_weight_merge, cluster_module
# from matching.permutation2 import permute_module

__all__ = ['weight_clustering']


# --------- new version of weight matching ----------------- #

def weight_clustering(model, device='cpu', pairing_rate=1.0, verbose=True, save_path=None, model_config_path=None, isllm=False, llm_recipe=None):
    print_verbose(verbose, 20*"-"+"perform weight clustering"+20*"-")
    model = copy.deepcopy(model).to(device)

    model_graph, final_module_name = load_model_graph(model_config_path, verbose=verbose)
    if isllm:
        assert model_graph['model.embed_tokens']['merge'] is not None, f"No merge find in model.embed_tokens. Please check the model_config_path."

    params = model.state_dict()
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

    # -------------create parameter_merge_map------------------------------#
    print_verbose(verbose, 20*"-"+f"compute merge for each module"+20*"-")
    for module_name, module in model.named_modules():
        if module_name == '' or not_go_insider(module_name) or isinstance(module, nn.Embedding) or module_name == final_module_name:
            continue
        if module_name not in model_graph:
            if not isinstance(module, (LlamaAttention, LlamaMLP)):
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
        
        print(f"processing {module_name_} with pr={pr}")
        # assert model_graph['model.embed_tokens']['merge'] is not None, f"No merge find in model.embed_tokens. Please check the model_config_path."

        if is_batchnorm(module) or isinstance(module, LlamaRMSNorm) or is_pool(module) or isinstance(module, LlamaRotaryEmbedding) or len(list(module.named_parameters()))==0:
            pre_name  = model_graph[module_name]['pre']
            model_graph[module_name]['merge'] = model_graph[pre_name]['merge']
            print_verbose(verbose, f'{module_name} reuses the same merge from {pre_name}. No folding on this layer.')
            continue

        if model_graph[module_name_]['consistent_map'] is not None:
            consistent_module_name = model_graph[module_name_]['consistent_map']
            assert model_graph[consistent_module_name]['merge'] is not None, f"No merge find in {consistent_module_name}. Please check the model_config_path."
            model_graph[module_name_]['merge'] = model_graph[consistent_module_name]['merge']
            print_verbose(verbose, f"To matain consistency in a residual block, we apply {consistent_module_name}'s merge on {module_name_}")
            continue

        if not model_graph[module_name_]['do_folding']:
            continue
        # assert model_graph['model.embed_tokens']['merge'] is not None, f"No merge find in model.embed_tokens. Please check the model_config_path."
        
        if save_path is not None:
            # pass
            merge_module_path = f'{save_path}/{round(pr, 2)}/{module_name}_merge_mtx.npy'
            if os.path.exists(merge_module_path):
                # if you can find existing merge, directly use it. Otherwise, we will compute it.
                existing_merge = torch.from_numpy(np.load(merge_module_path)).long()
                model_graph[module_name_]['merge'] = existing_merge
                print_verbose(verbose, f"Load existing merge from {merge_module_path} for {module_name_}")
                if isinstance(module, LlamaAttention):
                    o_submodule_name = f"{module_name}_o"
                    print(o_submodule_name)
                    assert model_graph[o_submodule_name]['consistent_map'] is not None, f"No consistent map find in {o_submodule_name}. Please check the model_config_path."
                    consistent_module_name = model_graph[o_submodule_name]['consistent_map']
                    assert model_graph[consistent_module_name]['merge'] is not None, f"No merge find in {consistent_module_name}. Please check the model_config_path."
                    model_graph[o_submodule_name]['merge'] = model_graph[consistent_module_name]['merge']
                    print_verbose(verbose, f"To matain consistency in a residual block, we apply {consistent_module_name}'s merge on {o_submodule_name}")
                if isinstance(module, LlamaMLP):
                    down_submodule_name = f"{module_name}_down"
                    print(down_submodule_name)
                    assert model_graph[down_submodule_name]['consistent_map'] is not None, f"No consistent map find in {down_submodule_name}. Please check the model_config_path."
                    consistent_module_name = model_graph[down_submodule_name]['consistent_map']
                    assert model_graph[consistent_module_name]['merge'] is not None, f"No merge find in {consistent_module_name}. Please check the model_config_path."
                    model_graph[down_submodule_name]['merge'] = model_graph[consistent_module_name]['merge']
                    print_verbose(verbose, f"To matain consistency in a residual block, we apply {consistent_module_name}'s merge on {down_submodule_name}")
                continue
        #-------------------------For each module, compute the merge of the output channel.-----------------------------#
        print_verbose(verbose, f"Compute merge on {module_name_}'s output channel:")
        n = model_graph[module_name_]['num_channels']
        order = "output"
        if isinstance(module, LlamaAttention):
            merge, inertia = get_weight_merge(module, module_name, model_graph, model, params, order, n=n, n_folded=round(n*(1-pr)), device=device, normalize=False, use_kmeans=False,verbose=verbose)
            model_graph[module_name_]['merge'] = merge
            no_go_inside_the_module.append(module_name) # not to go attention.q,k,v,o. We will go qkv and o separately.
            assert model_graph[f"{module_name}_o"]['consistent_map'] is not None, f"No consistent map find in {module_name}_o. Please check the model_config_path."
            consistent_module_name = model_graph[f"{module_name}_o"]['consistent_map']
            model_graph[f"{module_name}_o"]['merge'] = model_graph[consistent_module_name]['merge']
            print_verbose(verbose, f"To matain consistency in a residual block, we apply {consistent_module_name}'s merge on {module_name}_o")

        elif isinstance(module, LlamaMLP):
            no_go_inside_the_module.append(module_name) # not to go gate and up. We will go gate and up separately.
            merge, inertia = get_weight_merge(module, module_name, model_graph, model, params, order, n=n, n_folded=round(n*(1-pr)), device=device, normalize=False, use_kmeans=False,verbose=verbose)
            model_graph[module_name_]['merge'] = merge
            assert model_graph[f"{module_name}_down"]['consistent_map'] is not None, f"No consistent map find in {module_name}_down. Please check the model_config_path."
            consistent_module_name = model_graph[f"{module_name}_down"]['consistent_map']
            model_graph[f"{module_name}_down"]['merge'] = model_graph[consistent_module_name]['merge']
            print_verbose(verbose, f"To matain consistency in a residual block, we apply {consistent_module_name}'s merge on {module_name}_down")
        elif isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            merge, inertia = get_weight_merge(module, module_name, model_graph, model, params, order, n=n, n_folded=round(n*(1-pr)), device=device, normalize=False, use_kmeans=False,verbose=verbose)
            model_graph[module_name_]['merge'] = merge
        else:
            raise ValueError(f"We don't match this type of module: {module_name}")

        if save_path is not None:
            os.makedirs(f'{save_path}/{round(pr, 2)}', exist_ok=True)
            merge_module_path = f'{save_path}/{round(pr, 2)}/{module_name}_merge_mtx.npy'
            inertia_module_path = f'{save_path}/{round(pr, 2)}/{module_name}_merge_inertia.npy'
            print_verbose(verbose, f'Save merge in {merge_module_path}')
            print_verbose(verbose, f'Save inertia in {inertia_module_path}')
            print(f"inertia: {inertia}")
            np.save(merge_module_path, merge.detach().cpu().numpy())
            np.save(inertia_module_path, inertia)






    no_go_inside_the_module = []
    # -------------Another loop to merge the channels------------------------------#
    print_verbose(verbose, 20*"-"+f"Perform weight clustering and merge the channels"+20*"-")
    for module_name, module in model.named_modules():
        if module_name == '' or not_go_insider(module_name) or isinstance(module, nn.Embedding) or module_name == final_module_name:
            continue
        if module_name not in model_graph:
            if not isinstance(module, (LlamaAttention, LlamaMLP)):
                continue
        
        # For LlamaAttention and LlamaMLP, we need to adjust the module_name
        module_name_ = module_name
        if isinstance(module, LlamaAttention):
            module_name_ = f"{module_name}_qkv"
        elif isinstance(module, LlamaMLP):
            module_name_ = f"{module_name}_gateup"
    

        if isllm:
            if module_name in llm_recipe:
                # in llm_recipe, to debug easily, we only define pr for whole module not for submodules. #todo: merge folding_recipe and model_config before release
                pr = llm_recipe[module_name]['pr']
            else:
                continue
        else:
            pr = pairing_rate
        

        if is_pool(module) or isinstance(module, LlamaRotaryEmbedding) or len(list(module.named_parameters()))==0 or not model_graph[module_name_]['do_folding']:
            # model_graph[module_name_]['merge']
            print_verbose(verbose, f"No folding on {module_name_}")
            continue
        
        if isinstance(module, LlamaAttention):
            cluster_module(model, module, model_graph, module_name, order="output", device=device, verbose=verbose)
            no_go_inside_the_module.append(module_name) # not to go attention.q,k,v,o. We will go qkv and o separately.
        elif isinstance(module, LlamaMLP):
            cluster_module(model, module, model_graph, module_name, order="output", device=device, verbose=verbose)
            no_go_inside_the_module.append(module_name) # not to go gate and up. We will go gate and up separately.
        elif isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            cluster_module(model, module, model_graph, module_name, order="output", device=device, verbose=verbose)
        # elif isinstance(module, nn.BatchNorm2d):
        #     merge = model_graph[module_name_]['merge']
        #     cluster_module(model, module, model_graph, module_name, order="output", verbose=verbose)
        # elif isinstance(module, LlamaRMSNorm):
        #     merge = model_graph[module_name_]['merge']
        #     cluster_module(model, module, model_graph, module_name, order="output", verbose=verbose)
        # else:
            # raise ValueError(f"We don't merge this type of module: {module_name}")

    torch.cuda.empty_cache()
    return model, model_graph, final_module_name
