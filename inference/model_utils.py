# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoConfig, LlamaForCausalLM, LlamaConfig, GemmaForCausalLM, GemmaConfig, MistralForCausalLM, MistralConfig

# Function to load the main model for text generation
def load_model(model_name, quantization):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return model


# Function to load the PeftModel for performance optimization
def load_peft_model(model, peft_model):
    peft_model = PeftModel.from_pretrained(model, peft_model)
    return peft_model

# Loading the model from config to load FSDP checkpoints into that
def load_model_from_config(config_path):
    if "llama" in config_path.lower():
        model_config = LlamaConfig.from_pretrained(config_path)
        model = LlamaForCausalLM(config=model_config)
    elif "gemma" in config_path.lower():
        model_config = GemmaConfig.from_pretrained(config_path)
        model = GemmaForCausalLM(config=model_config)
    elif "mistral" in config_path.lower():
        model_config = MistralConfig.from_pretrained(config_path) 
        model = MistralForCausalLM(config=model_config)
    else:
        raise NotImplementedError()
    return model
    
    