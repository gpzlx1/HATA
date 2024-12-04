import json
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from fastchat.model import get_conversation_template
from transformers.generation.configuration_utils import GenerationConfig


def longchat_appy_chat_template(prompt, tokenizer):
    conv = get_conversation_template("vicuna")
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    encoded = tokenizer(prompt)
    return encoded


def mistral_apply_chat_template(prompt, tokenizer):
    prompt = f"[INST] {prompt} [/INST]"
    encoded = tokenizer(prompt)
    return encoded


def llama2_apply_chat_template(prompt, tokenizer):
    prompt = f"[INST] {prompt} [/INST]"
    encoded = tokenizer(prompt)
    return encoded


def llama3_apply_chat_template(prompt, tokenizer):
    messages = [{"role": "user", "content": f"{prompt}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           add_generation_prompt=True,
                                           tokenize=False)
    encoded = tokenizer(prompt)
    return encoded


def llama_load_model_and_tokenizer(args, model_name_or_path, **kwargs):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    generate_kwarg = {}

    method = args.method.lower()

    if method == "flashinfer":
        generate_config = {
            "page_num": 1000,
            "page_size": 16,
        }
        from myTransformer.models.modeling_llama import CustomLlamaForCausalLM
        model = CustomLlamaForCausalLM.from_pretrained(model_name_or_path,
                                                       config=model_config)

    elif method == "flashattn":
        generate_config = {
            "max_gpu_cache_memory": 25 * 1024 * 1024 * 1024,  # 30GB
        }
        model_config._attn_implementation = "flash_attention_2"
        from myTransformer.models.modeling_llama_fa import CustomLlamaForCausalLM
        model = CustomLlamaForCausalLM.from_pretrained(model_name_or_path,
                                                       config=model_config)

    elif method == "hash":
        generate_config = {
            "max_gpu_cache_memory": 25 * 1024 * 1024 * 1024,
            "hash_rbits": 128,
            "hash_weights_path":
            "/root/workspace/myoffloading/KVOffloading/model_weights/longchat-7b-v1.5-32k-128",
            "sparse_ratio": 0.1,
        }
        model_config._attn_implementation = "sdpa"
        from myTransformer.models.modeling_llama_hash import CustomLlamaForCausalLM
        model = CustomLlamaForCausalLM.from_pretrained(model_name_or_path,
                                                       config=model_config)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                              fast_tokenizer=True,
                                              use_fast=True)

    # select apply_chat_template
    if any([
            x in model_name_or_path.lower()
            for x in ["llama-2", "llama2", "llama_2"]
    ]):
        print("run llama2 model")
        apply_chat_template = llama2_apply_chat_template
        tokenizer.pad_token = "[PAD]"
        tokenizer.padding_side = "left"

    elif any([
            x in model_name_or_path.lower() for x in
        ["llama-3.1", "llama3.1", "llama_3.1", "llama-3", "llama3", "llama_3"]
    ]):
        print("run llama3 model")
        apply_chat_template = llama3_apply_chat_template
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    elif any([x in model_name_or_path.lower() for x in ["mistral"]]):
        print("run mistral model")
        apply_chat_template = mistral_apply_chat_template
        tokenizer.pad_token = "[PAD]"
        tokenizer.padding_side = "left"

    elif any([x in model_name_or_path.lower() for x in ["longchat"]]):
        print("run longchat model")
        apply_chat_template = longchat_appy_chat_template

    else:
        raise ValueError("Unsupported model name")

    # unset to avoid some warning
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model = model.to(torch.float16).eval()

    for key, value in generate_config.items():
        setattr(model.generation_config, key, value)

    return model, tokenizer, generate_kwarg, apply_chat_template


def comm_generate(x, generate_kwarg, model, tokenizer):
    input_length = x["input_ids"].shape[1]

    output = model.generate(**x, do_sample=False, **generate_kwarg)

    output = output[:, input_length:]
    preds = tokenizer.batch_decode(output, skip_special_tokens=True)

    return preds
