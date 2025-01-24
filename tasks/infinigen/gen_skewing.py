import sys
import tqdm
import torch
import os
import argparse
from transformers import AutoConfig, AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
import numpy as np
import random

sys.path.append("../")
from llama_utils import get_model_type_arch


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def set_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model",
        type=str,
        default="/nfs/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--prompt_file", default="pg19_firstbook.txt")
    parser.add_argument("--max_context_length", type=int, default=131768)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pp_num", type=int, default=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_args(parser)
    args = parser.parse_args()
    seed_everything(42)

    model_name, model_arch = get_model_type_arch(args.model)

    if model_arch == "glm":
        tokenizer = AutoTokenizer.from_pretrained(args.model,
                                                  trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model,
                                                  fast_tokenizer=True,
                                                  use_fast=True)

    model_config = AutoConfig.from_pretrained(args.model,
                                              trust_remote_code=True)
    model_config._attn_implementation = "flash_attention_2"
    model_config.torch_dtype = torch.float16
    if model_arch == "glm":
        from modeling_glm_fa_save_qk import CustomGlmForCausalLM
        model = CustomGlmForCausalLM.from_pretrained(args.model,
                                                     torch_dtype=torch.float16,
                                                     config=model_config,
                                                     trust_remote_code=True)
    elif model_arch == "llama":
        from modeling_llama_fa_save_qk import CustomLlamaForCausalLM
        model = CustomLlamaForCausalLM.from_pretrained(args.model,
                                                       config=model_config)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model = model.to(torch.float16).eval()

    device_ids = [i for i in range(args.pp_num)]
    if len(device_ids) > 1:
        max_memory = {}
        for id in device_ids:
            max_memory[id] = torch.cuda.mem_get_info(id)[0]
        map_kwargs = {"max_memory": get_balanced_memory(model, max_memory)}
        device_map = infer_auto_device_map(model,
                                           no_split_module_classes=[
                                               "CustomLlamaDecoderLayer",
                                               "CustomGLMBlock",
                                           ],
                                           **map_kwargs)
        model = dispatch_model(model, device_map=device_map)
    else:
        model = model.to(device_ids[0])

    generation_kwargs = {
        "max_gpu_cache_memory": 0.0 * 1024 * 1024 * 1024,
    }
    generation_config = GenerationConfig(**generation_kwargs)

    num_layers = model_config.num_hidden_layers
    if model_arch == "glm":
        num_heads = model_config.num_attention_heads
        num_kv_heads = model_config.multi_query_group_num
    else:
        num_heads = model_config.num_attention_heads
        num_kv_heads = (model_config.num_attention_heads if getattr(
            model_config, "num_key_value_heads", None) is None else
                        model_config.num_key_value_heads)
    head_dim = (model_config.head_dim if hasattr(model_config, "head_dim") else
                model_config.hidden_size // model_config.num_attention_heads)
    gqa_size = num_heads // num_kv_heads

    querys = []
    keys = []

    with open(args.prompt_file, 'r') as file:
        prompt = file.read()
    encoded_data = tokenizer(prompt, return_tensors="pt")
    input = {
        "input_ids":
        encoded_data["input_ids"][:, :args.max_context_length].cuda(),
        "attention_mask":
        encoded_data["attention_mask"][:, :args.max_context_length].cuda()
    }

    model.generate(**input,
                   do_sample=False,
                   min_new_tokens=1,
                   max_new_tokens=1,
                   generation_config=generation_config)
    for l in range(num_layers):
        if model_arch == "glm":
            query = model.transformer.encoder.layers[
                l].self_attention.post_rotary_query.cpu()
            key = model.transformer.encoder.layers[
                l].self_attention.post_rotary_key.cpu()
        else:
            query = model.model.layers[l].self_attn.post_rotary_query.cpu()
            key = model.model.layers[l].self_attn.post_rotary_key.cpu()
        querys.append(query)
        keys.append(key)

    del model

    os.makedirs(args.output_dir, exist_ok=True)

    for l in tqdm.tqdm(range(num_layers), desc="Run svd on layers"):
        skewing_matrix = torch.zeros((num_kv_heads, head_dim, head_dim),
                                     device="cpu")

        query = querys[l]
        key = keys[l]
        if gqa_size > 1:
            query = query.view(-1, num_kv_heads, gqa_size, head_dim)
            query = torch.mean(query, dim=2)

        for head in range(num_kv_heads):
            in_q = query[:, head, :]
            in_k = key[:, head, :]
            uq, sq, vq = torch.svd(in_q.to(torch.float))
            uk, sk, vk = torch.svd(in_k.to(torch.float))
            s = sq * sk
            a = torch.zeros(head_dim, head_dim, device="cpu")
            _, ind = s.sort()
            r, c = a.shape
            skewing_matrix[head] = a.scatter(-1,
                                             ind.unsqueeze(0).repeat(r, 1),
                                             vq).to(torch.float16)

        torch.save(skewing_matrix,
                   os.path.join(args.output_dir, f"skewing_martix_{l:02d}.pt"))
