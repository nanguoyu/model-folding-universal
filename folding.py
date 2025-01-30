import time
import torch
import numpy as np
from torch import nn
import pandas as pd
import copy
import os
from train import full_eval, evaluate2
import argparse
import models
from models import *
from cvdataset import load_data, dataset_infor
import wandb
from utils import compare_models, local_structured_prune_model, local_unstructured_prune_model, count_parameters
from core.utils import has_batchnorm2d, has_batchnorm1d
from core.repair import repair
import core
from core.repair import reset_bn_stats
from llmdata import text_datasets, get_loaders
from transformers import AutoTokenizer
from llmeval import eval_ppl, interactive_eval
from utils import print_verbose
from llama_2_7b_hf_recipe import llama_recipe as llama_recipe_2_7b
from llama_1_7b_hf_recipe import llama_recipe as llama_recipe_1_7b
from llama_3_2_1b_hf_recipe import llama_recipe as llama_recipe_3_2_1b
torch.cuda.empty_cache()


llm_sparsity_recipe = {
    'llama_1_7b_hf': llama_recipe_1_7b,
    'llama_2_7b_hf': llama_recipe_2_7b,
    'llama_3_2_1b_hf': llama_recipe_3_2_1b,
}


def repair_fun(model1, model2, model_mix_perm, train_loader, device, epochs=1, alpha=0.5):
    if has_batchnorm2d(model_mix_perm) or has_batchnorm1d(model_mix_perm):
        print(f"Found batchorm in this model")
        print('Apply REPAIR(reset bn)')
        model_mix_repair = reset_bn_stats(model_mix_perm, loader=train_loader, device=device, epochs=epochs)
    else:
        print(f"Not find batchorm in this model")
        print('Apply REPAIR')
        print("*"*10)
        model_mix_repair = repair(model1, model2, model_mix=model_mix_perm.to(device), loader=train_loader, device=device, alpha=alpha)
    return model_mix_repair

def main():
    model_names = sorted(name for name in models.__dict__
                        if name.islower()
                        and callable(models.__dict__[name]))
    
    matching_methods = sorted(name for name in core.__dict__
                     if name.islower() and name.endswith('matching')
                    and callable(core.__dict__[name]))
    parser = argparse.ArgumentParser(description='Model Folding')
    parser.add_argument('--result_path', default='result', type=str)
    parser.add_argument('--dataset', default='FashionMNIST', type=str, help='FashionMNIST')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--model', '-m', metavar='MODEL', default='lenet5',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: lenet5)')
    parser.add_argument('--wider_factor', default=1, type=int)
    parser.add_argument('--gpus', default=0, type=int)
    parser.add_argument('--datadir', default='datasets', type=str)
    parser.add_argument('--weight', default='./weights/lenet5_4Xwider.pth', type=str)
    parser.add_argument('--model_perm_config_path', default='./config/llama_2_7b_hf_perm.json', type=str)
    parser.add_argument('--matching_method', default='weight_clustering', type=str,
                        choices=matching_methods,
                        help='matching methods: ' +
                            ' | '.join(matching_methods) +
                            ' (default:activation_matching)')
    parser.add_argument('--zero_shot_eval', action='store_true', help='evaluate zero-shot performance')
    parser.add_argument('--interactive_eval', action='store_true', help='interactive evaluation')
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()
    model_profile_str =  f'{args.model}_{args.wider_factor}Xwider_{args.dataset}' if args.wider_factor>1 else f'{args.model}_{args.dataset}'
    wandb_proiject_name = "model folding" 
    wandb.init(project=wandb_proiject_name, name=f'folding_{model_profile_str}_{args.matching_method}', entity="naguoyu", 
               config={"dataset":args.dataset, 'model':model_profile_str , 'matching':args.matching_method},)
    print(f'Folding model:{model_profile_str}')
    os.makedirs(args.result_path, exist_ok=True)

    if args.gpus==-1:
        device= torch.device('cpu')
    else:
        device = torch.device(f"cuda:{args.gpus}" if torch.cuda.is_available() else torch.device('cpu') )
    # torch.cuda.set_device(device) # new tensor will be allocated on the specified device
    print(f'using device {device}')
    
    time_start = time.time()
    
    # -------handle dataset specific information-------#
    if args.dataset in dataset_infor.keys():
        nchannels = dataset_infor[args.dataset]['num_channels']
        num_classes = dataset_infor[args.dataset]['num_classes']
    elif args.dataset in text_datasets:
        nchannels = None
        num_classes = None
    else:
        raise ValueError(f"{args.dataset} is not supported now. ")
    
    if args.model in llm_models:
        model_name = llm_models_hf_path[args.model]
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    else:
        model_name = args.model
    model1 = models.__dict__[args.model](
                                        wider_factor=args.wider_factor,
                                        n_channels = nchannels,
                                        num_classes= num_classes,
                                        weights=args.weight,
                                        )
    model1.eval()
    model1.to(device)


    # --------------------------- init dataset---------------------------#
    if args.dataset in text_datasets:
        # LLM dataset
        print(f"Using text dataset: {args.dataset}")
        train_loader, test_loader = get_loaders(args.dataset, seed=0, seqlen=model1.seqlen, tokenizer=tokenizer)
    else:
        print(f"Using computer vision dataset: {args.dataset}")
        # Computer vision dataset
        train_loader = load_data("train", args.dataset, datadir=args.datadir, nchannels=nchannels, batch_size=args.batch_size, shuffle=True,device=device,num_workers=8)
        test_loader = load_data("test", args.dataset, datadir=args.datadir, nchannels=nchannels, batch_size=args.batch_size, shuffle=True,device=device,num_workers=8)

    if args.model in llm_models:
        llm_recipe = llm_sparsity_recipe[args.model]
    else:
        llm_recipe = None
    columns = ['sparsity', 'folding w REPAIR', 'folding w/o REPAIR']
    results = []

    # pr_array_for_plot_in_paper = list(range(50, 51, 10))
    pr_array_for_plot_in_paper = list(range(10, 100, 10))

    time_before_loop = time.time()
    print(f"Time used to process before loop: {time_before_loop - time_start}")
    for pr in pr_array_for_plot_in_paper:
    # for pr in range(50, 51, 1):
    # for pr in range(40, 41, 1):
    # for pr in range(100, 101, 1):
    # for pr in range(20, 21, 1):
        time_loop_start = time.time()

        pr *=0.01

        print("="*40+f"Pairing rate = {pr}"+"="*40)

        os.makedirs(f'{args.result_path}/{model_profile_str}', exist_ok=True)

        save_path = f'{args.result_path}/{model_profile_str}/weight_matching'
        os.makedirs(save_path, exist_ok=True)
        # model2, model_graph, final_module_name = matching.weight_matching(model1, device=device, pairing_rate=pr, 
        #                                                                             save_path=save_path, distance_metric="scalar",
        #                                                                             verbose=args.verbose,
        #                                                                             model_config_path=args.model_perm_config_path,
        #                                                                             isllm=args.model in llm_models,
        #                                                                             llm_recipe=llm_recipe)
        model2, model_graph, final_module_name = core.weight_clustering(model1, device=device, pairing_rate=pr, 
                                                                                    save_path=save_path,
                                                                                    verbose=args.verbose,
                                                                                    model_config_path=args.model_perm_config_path,
                                                                                    isllm=args.model in llm_models,
                                                                                    llm_recipe=llm_recipe)

        flag, msg = compare_models(model1, model2)
        print_verbose(args.verbose, f'After folding, are model1 and model2 the same model : {flag} {msg}')


        num_params_original_model = count_parameters(model1)
        num_params_folded_model = count_parameters(model2)
        print(f"num of params in original model: {num_params_original_model}")
        print(f"num of params in folded model: {num_params_folded_model}")

        parameter_sparsity = 1-num_params_folded_model/num_params_original_model
        print(f"sparsity:{parameter_sparsity}")
        

        if args.model in llm_models:
            print("Original model's performance:")
            # ppl_test = eval_ppl(model1.to(device), tokenizer, device=device)
            # print(f"wikitext2 perplexity of original model {ppl_test}")
            print("Folded model's performance:")
            ppl_test = eval_ppl(model2.to(device), tokenizer, device=device)
            print(f"\twikitext2 perplexity of folded model {ppl_test}")
            print("LLM model, no need to repair.")
            if args.interactive_eval:
                interactive_eval(model2, tokenizer, device)
            if args.zero_shot_eval:
                print("Evaluating zero-shot performance")
                from llmeval import eval_zero_shot
                zero_shot_result = eval_zero_shot(model_name, model2, tokenizer, task_list=["boolq","winogrande","arc_challenge","arc_easy"], num_fewshot=0, use_accelerate=False, add_special_tokens=False)
                print("********************************\nzero_shot evaluation results\n********************************")
                print(zero_shot_result)
            print("Done")
            return
        else:
            print("Original model's performance:")
            evaluate2(model1.to(device), test_loader, device=device, verbose=True)
            print("Folded model's performance:")
            folding_test_acc, _ = evaluate2(model2.to(device), test_loader, device=device, verbose=True)
            # Apply REPAIR to the folded model
            model_repaired = repair_fun(model1, model1, model2.to(device), train_loader, device=device, epochs=1, alpha=0.5)

            time_finish_folding = time.time()
            print(f"Time used to process in folding {pr*100:.0f}%: {time_finish_folding - time_loop_start} seconds")
            print('model performance after folding+repair')
            folding_test_acc_repair, _ = evaluate2(model_repaired.to(device), test_loader, device=device, verbose=True)




        wandb.log({
                "sparsity":round(parameter_sparsity, 2),
                "folding/test accuracy":round(folding_test_acc/100, 2),
                    "folding/test accuracy REPAIR":round(folding_test_acc_repair/100, 2),
                    })
        results.append([round(parameter_sparsity*100, 4), round(folding_test_acc_repair/100, 2), round(folding_test_acc/100, 2)])

    df = pd.DataFrame(results, columns=columns)

    df.to_csv(f'result/{model_profile_str}/test_acc_result_folding.csv', index=False)
    print(f"Result saved to result/{model_profile_str}/test_acc_result_folding.csv")
    print("Done")

if __name__ == '__main__':
    main()
# ResNet18
# CUDA_VISIBLE_DEVICES=2 python folding.py  --gpus=0 --dataset=CIFAR10 --wider_factor=1 --weight='./weights/resnet18_CIFAR10.pth' --model=resnet18 --model_perm_config_path='./config/resnet18_perm.json'

# llama1
# CUDA_VISIBLE_DEVICES=4 python folding.py  --gpus=0 --dataset=wikitext2  --model=llama_1_7b_hf --model_perm_config_path='./config/llama_1_7b_hf_perm.json' --zero_shot_eval
# llama2
# CUDA_VISIBLE_DEVICES=3 python folding.py  --gpus=0 --dataset=wikitext2  --model=llama_2_7b_hf --model_perm_config_path='./config/llama_2_7b_hf_perm.json' --zero_shot_eval
# llama3.2
# CUDA_VISIBLE_DEVICES=3 python folding.py  --gpus=0 --dataset=wikitext2  --model=llama_3_2_1b_hf --model_perm_config_path='./config/llama_3_2_1b_hf_perm.json' --zero_shot_eval