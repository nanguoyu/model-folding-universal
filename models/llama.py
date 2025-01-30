from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP

from torch import nn
import torch
from torch import Tensor
from typing import Any
from functools import wraps
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

__all__ = ['llama_2_7b_hf', 'llama_1_7b_hf', 'llama_3_2_1b_hf']

def clean_llama_kwargs(func):
    @wraps(func)
    def wrapper(**kwargs):
        valid_kwargs = []
        
        extra_kwargs = {k: kwargs[k] for k in kwargs if k not in valid_kwargs}
        if extra_kwargs:
            print(f"Removed unsupported kwargs: {list(extra_kwargs.keys())}")
            kwargs = {k: kwargs[k] for k in kwargs if k in valid_kwargs}
        return func(**kwargs)
    return wrapper

@clean_llama_kwargs
def llama_2_7b_hf(**kwargs: Any) -> LlamaForCausalLM:
    model =  LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path="meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
        )
    model.seqlen = model.config.max_position_embeddings
    return model


@clean_llama_kwargs
def llama_1_7b_hf(**kwargs: Any) -> LlamaForCausalLM:
    model =  LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path="huggyllama/llama-7b",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
        )
    model.seqlen = model.config.max_position_embeddings

    return model


@clean_llama_kwargs
def llama_3_2_1b_hf(**kwargs: Any) -> LlamaForCausalLM:
    model =  LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-1B",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
        )
    # model.seqlen = model.config.max_position_embeddings
    model.seqlen = 2048
    return model