import os
import torch
import transformers
from torch import bfloat16
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM,AutoTokenizer
import re
def setup_model_tokenizer(is_quantized= True, model_name=None):
    if model_name is None:
        model_name = os.environ.get("MODEL_NAME")
    
    if re.search("llama", model_name):

        if is_quantized:
            bnb_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=bfloat16
            )
            model = LlamaForCausalLM.from_pretrained(model_name, output_hidden_states=True,
                                                    quantization_config=bnb_config, device_map=os.environ.get('CUDA_CORE'))

        else:
            model =  LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf',output_hidden_states=True,torch_dtype=torch.float16,device_map = "auto").half()

        tokenizer = LlamaTokenizer.from_pretrained(model_name, add_special_tokens=False,
                                                add_bos_token=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True,
                                                    device_map="auto")

        tokenizer = AutoTokenizer.from_pretrained(model_name, add_special_tokens=False,
                                                add_bos_token=False)

    return tokenizer, model


def take_down_model(model):
    del model
    torch.cuda.empty_cache()