import torch
from transformers import AutoConfig, AutoTokenizer
import os


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


def qwen2_apply_chat_template(prompt, tokenizer):
    messages = [{"role": "user", "content": f"{prompt}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           add_generation_prompt=True,
                                           tokenize=False)
    encoded = tokenizer(prompt)
    return encoded


def get_model_type_arch(model_name_or_path):
    if any([
            x in model_name_or_path.lower()
            for x in ["llama-2", "llama2", "llama_2"]
    ]):
        print("run llama2 model")
        return "llama2", "llama"
    elif any([
            x in model_name_or_path.lower() for x in
        ["llama-3.1", "llama3.1", "llama_3.1", "llama-3", "llama3", "llama_3"]
    ]):
        print("run llama3 model")
        return "llama3", "llama"
    elif any([x in model_name_or_path.lower() for x in ["qwen2", "qwen2.5"]]):
        print("run qwen2 model")
        return "qwen2", "qwen2"
    else:
        raise ValueError("Unsupported model name")


def load_config_and_tokenizer(args, model_name_or_path):
    dtype = torch.float16
    model_config = AutoConfig.from_pretrained(model_name_or_path,
                                              trust_remote_code=True)
    model_config._attn_implementation = "flash_attention_2"
    model_config.torch_dtype = dtype
    generate_kwargs = {}

    method = args.method.lower()

    model_type, model_arch = get_model_type_arch(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                              fast_tokenizer=True,
                                              use_fast=True)

    # select apply_chat_template
    if model_type == "llama2":
        apply_chat_template = llama2_apply_chat_template
        tokenizer.pad_token = "[PAD]"
        tokenizer.padding_side = "left"

    elif model_type == "llama3":
        print("run llama3 model")
        apply_chat_template = llama3_apply_chat_template
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    elif model_type == "qwen2":
        apply_chat_template = qwen2_apply_chat_template

    else:
        raise ValueError("Unsupported model name")

    if method == "flashattn":
        generate_config = {
            "max_gpu_cache_memory":
            float(os.environ["CUDA_MEM"]) * 1024 * 1024 * 1024,  # 30GB
        }

    elif method == "hash":
        generate_config = {
            "max_gpu_cache_memory":
            float(os.environ["CUDA_MEM"]) * 1024 * 1024 * 1024,
            "hash_rbits": int(os.environ["RBIT"]),
            "hash_weights_path": os.environ["HASH_WEIGHTS_PATH"],
            # "hash_weights_path": None,
            "sparse_ratio": float(os.environ["TOPK_RATIO"]),
            "use_norm": int(os.environ["USE_NORM"]) > 0,
            "with_bias": False,
            "num_sink": int(os.environ["NUM_SINK"]),
            "num_recent": int(os.environ["NUM_RECENT"]),
        }

    else:
        raise ValueError(f"Unsupported method: {method}")

    model_meta = (method, model_arch, generate_config)

    return model_meta, model_config, tokenizer, generate_kwargs, apply_chat_template


def load_model(model_meta, model_config, model_name_or_path):
    method, model_arch, generate_config = model_meta

    if method == "flashattn":
        if model_arch == "llama":
            from myTransformer.models.llama.modeling_llama_fa import CustomLlamaForCausalLM
            model = CustomLlamaForCausalLM.from_pretrained(model_name_or_path,
                                                           config=model_config)

        elif model_arch == "qwen2":
            from myTransformer.models.qwen2.modeling_qwen2_fa import CustomQwen2ForCausalLM
            model = CustomQwen2ForCausalLM.from_pretrained(model_name_or_path,
                                                           config=model_config)
        else:
            raise NotImplementedError(
                f"{method} not implemented for {model_arch} models!")

    elif method == "hash":
        if model_arch == "llama":
            from myTransformer.models.llama.modeling_llama_multi_hash import CustomLlamaForCausalLM
            model = CustomLlamaForCausalLM.from_pretrained(model_name_or_path,
                                                           config=model_config)

        elif model_arch == "qwen2":
            from myTransformer.models.qwen2.modeling_qwen2_multi_hash import CustomQwen2ForCausalLM
            model = CustomQwen2ForCausalLM.from_pretrained(model_name_or_path,
                                                           config=model_config)
        else:
            raise NotImplementedError(
                f"{method} not implemented for {model_arch} models!")

    else:
        raise ValueError(f"Unsupported method: {method}")

    # unset to avoid some warning
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    dtype = torch.float16
    model = model.to(dtype).eval()

    # setup model
    for key, value in generate_config.items():
        setattr(model.generation_config, key, value)

    return model


def llama_load_model_and_tokenizer(args, model_name_or_path, **kwargs):
    dtype = torch.float16
    model_config = AutoConfig.from_pretrained(model_name_or_path,
                                              trust_remote_code=True)
    model_config.torch_dtype = dtype
    generate_kwarg = {}

    method = args.method.lower()

    model_type, model_arch = get_model_type_arch(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                              fast_tokenizer=True,
                                              use_fast=True)

    # select apply_chat_template
    if model_type == "llama2":
        apply_chat_template = llama2_apply_chat_template
        tokenizer.pad_token = "[PAD]"
        tokenizer.padding_side = "left"

    elif model_type == "llama3":
        print("run llama3 model")
        apply_chat_template = llama3_apply_chat_template
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    elif model_type == "qwen2":
        apply_chat_template = qwen2_apply_chat_template

    else:
        raise ValueError("Unsupported model name")

    if method == "flashattn":
        generate_config = {
            "max_gpu_cache_memory":
            float(os.environ["CUDA_MEM"]) * 1024 * 1024 * 1024,  # 30GB
        }
        model_config._attn_implementation = "flash_attention_2"
        if model_arch == "llama":
            from myTransformer.models.llama.modeling_llama_fa import CustomLlamaForCausalLM
            model = CustomLlamaForCausalLM.from_pretrained(model_name_or_path,
                                                           config=model_config)

        elif model_arch == "qwen2":
            from myTransformer.models.qwen2.modeling_qwen2_fa import CustomQwen2ForCausalLM
            model = CustomQwen2ForCausalLM.from_pretrained(model_name_or_path,
                                                           config=model_config)
        else:
            raise NotImplementedError(
                f"{method} not implemented for {model_arch} models!")

    elif method == "hash":
        generate_config = {
            "max_gpu_cache_memory":
            float(os.environ["CUDA_MEM"]) * 1024 * 1024 * 1024,
            "hash_rbits": int(os.environ["RBIT"]),
            "hash_weights_path": os.environ["HASH_WEIGHTS_PATH"],
            # "hash_weights_path": None,
            "sparse_ratio": float(os.environ["TOPK_RATIO"]),
            "use_norm": int(os.environ["USE_NORM"]) > 0,
            "with_bias": False,
            "num_sink": int(os.environ["NUM_SINK"]),
            "num_recent": int(os.environ["NUM_RECENT"]),
        }
        model_config._attn_implementation = "flash_attention_2"
        if model_arch == "llama":
            from myTransformer.models.llama.modeling_llama_multi_hash import CustomLlamaForCausalLM
            model = CustomLlamaForCausalLM.from_pretrained(model_name_or_path,
                                                           config=model_config)

        elif model_arch == "qwen2":
            from myTransformer.models.qwen2.modeling_qwen2_multi_hash import CustomQwen2ForCausalLM
            model = CustomQwen2ForCausalLM.from_pretrained(model_name_or_path,
                                                           config=model_config)
        else:
            raise NotImplementedError(
                f"{method} not implemented for {model_arch} models!")

    else:
        raise ValueError(f"Unsupported method: {method}")

    # unset to avoid some warning
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model = model.to(dtype).eval()

    for key, value in generate_config.items():
        setattr(model.generation_config, key, value)

    return model, tokenizer, generate_kwarg, apply_chat_template


def comm_generate(x, generate_kwarg, model, tokenizer):
    input_length = x["input_ids"].shape[1]

    output = model.generate(**x, do_sample=False, **generate_kwarg)

    output = output[:, input_length:]

    preds = tokenizer.batch_decode(output, skip_special_tokens=True)

    return preds
