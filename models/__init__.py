from .lenet import *
from .vgg import *
from .resnet import *
from .llama import llama_2_7b_hf, llama_1_7b_hf, llama_3_2_1b_hf
from .mlp import *
llm_models = ['llama_2_7b_hf', 'llama_1_7b_hf', 'llama_3_2_1b_hf']

llm_models_hf_path = {
    'llama_2_7b_hf': "meta-llama/Llama-2-7b-hf",
    'llama_1_7b_hf': "huggyllama/llama-7b",
    'llama_3_2_1b_hf': "meta-llama/Llama-3.2-1B",
}