import torch
from transformers import AutoTokenizer, AutoConfig
import myTransformer
import json
from datasets import Dataset
from fastchat.model import get_conversation_template
from transformers.generation.configuration_utils import GenerationConfig
import argparse
import os
import sys
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
import random
import numpy as np

sys.path.append("../")

from dataloader import datasets_prompt
from llama_utils import get_model_type_arch

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model",
        type=str,
        default="/nfs/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--max_context_length", type=int, default=131072)
    parser.add_argument("--save_path",
                        type=str,
                        default="/mnt/ramdisk/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--longbench_path",
                        type=str,
                        default="/nfs/shared_LLM_dataset/LongBench/data")
    parser.add_argument("--longbench_v2_path",
                        type=str,
                        default="/nfs/shared_LLM_dataset/LongBench-v2")
    parser.add_argument("--pp_num", type=int, default=1)
    parser.add_argument("--apply_template", default=False, action="store_true")
    parser.add_argument("--pos_sample_ratio", type=float, default=0.1)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def apply_template(prompt, model_name, tokenizer):
    # llama3.1 & glm
    if model_name in ["llama3", "glm"]:
        messages = [{"role": "user", "content": f"{prompt}"}]
        prompt = tokenizer.apply_chat_template(messages,
                                               add_generation_prompt=True,
                                               tokenize=False)
    # llama2-instruct
    elif model_name == "llama2":
        prompt = f"[INST] {prompt} [/INST]"
    # longchat
    elif model_name == "longchat":
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    return prompt


def load_longbench_dataset(path, data_name):
    fin = open(os.path.join(path, f"{data_name}.jsonl"), "r", encoding="utf-8")
    lines = fin.readlines()
    fin.close()
    ret = []
    for line in lines:
        eg = json.loads(line)
        ret.append(eg)
    return ret


def load_longbench(path, code_num=3, eng_text_num=5, chn_text_num=3):
    print("LongBench")

    # chinese text
    lsht = load_longbench_dataset(path, "lsht")
    print(f"#Total lsht data: {len(lsht)}, pick {chn_text_num}")
    random.shuffle(lsht)
    lsht = lsht[:chn_text_num]
    for i, item in enumerate(lsht):
        lsht[i] = datasets_prompt["lsht"].format(input=item["input"],
                                                 context=item["context"])

    # english text
    gov_report_e = load_longbench_dataset(path, "qasper_e")
    print(f"#Total qasper_e data: {len(gov_report_e)}, pick {eng_text_num}")
    random.shuffle(gov_report_e)
    gov_report_e = gov_report_e[:eng_text_num]
    for i, item in enumerate(gov_report_e):
        gov_report_e[i] = datasets_prompt["qasper"].format(
            input=item["input"], context=item["context"])

    # code text
    repobenchp_e = load_longbench_dataset(path, "repobench-p_e")
    print(f"#Total repobench-p_e data: {len(repobenchp_e)}, pick {code_num}")
    random.shuffle(repobenchp_e)
    repobenchp_e = repobenchp_e[:code_num]
    for i, item in enumerate(repobenchp_e):
        repobenchp_e[i] = datasets_prompt["repobench-p"].format(
            input=item["input"], context=item["context"])

    return gov_report_e + lsht + repobenchp_e


def load_longbench_v2(path, code_num=1, text_num=1):
    # _id, domain, sub_domain, difficulty, length, question, choice_A, choice_B, choice_C, choice_D, answer, context
    data = json.load(
        open(os.path.join(path, 'data.json'), 'r', encoding='utf-8'))
    code_data = []
    text_data = []
    for item in data:
        if item["length"] != "long":
            continue
        domain, sub_domain = item["domain"], item["sub_domain"]
        if domain == "Code Repository Understanding":
            code_data.append(item)
        elif domain in ["Multi-Document QA", "Single-Document QA"]:
            text_data.append(item)
    print("LongBench-v2")
    print(f"#Total long code data: {len(code_data)}, pick {code_num}")
    print(f"#Total long Doc QA data: {len(text_data)}, pick {text_num}")
    random.shuffle(code_data)
    random.shuffle(text_data)
    ret = code_data[:code_num] + text_data[:text_num]
    prompt_template = datasets_prompt["longbench-v2"]
    for i, item in enumerate(ret):
        ret[i] = prompt_template.format(
            context=ret[i]["context"],
            question=ret[i]["question"],
            C_A=ret[i]["choice_A"],
            C_B=ret[i]["choice_B"],
            C_C=ret[i]["choice_C"],
            C_D=ret[i]["choice_D"],
        )
    return ret


def load_processed_longbench_v2(path):
    fin = open(
        "/nfs/shared_LLM_dataset/RULER/Meta-Llama-3.1-8B-Instruct/128K/niah_multivalue/validation.jsonl",
        "r",
        encoding="utf-8")
    lines = fin.readlines()
    fin.close()
    ret = []
    for line in lines:
        eg = json.loads(line)
        ret.append(eg["input"])

    return ret


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

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.torch_dtype = torch.bfloat16
    config._attn_implementation = "flash_attention_2"
    if model_arch == "glm":
        from myTransformer.models.glm.modeling_glm_fa_v2_build_dataset import CustomGlmForCausalLM
        model = CustomGlmForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            config=config,
            trust_remote_code=True)
    elif model_arch == "llama":
        from myTransformer.models.llama.modeling_llama_fa_build_dataset import CustomLlamaForCausalLM
        model = CustomLlamaForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, config=config)
        if model_name == "llama3":
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.padding_side = "left"
        elif model_name == "llama2":
            tokenizer.pad_token = "[PAD]"
            tokenizer.padding_side = "left"
    elif model_arch == "qwen2":
        from myTransformer.models.qwen2.modeling_qwen2_fa_build_dataset import CustomQwen2ForCausalLM
        model = CustomQwen2ForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, config=config)

    device_ids = [i for i in range(args.pp_num)]
    if len(device_ids) > 1:
        max_memory = {}
        for id in device_ids:
            max_memory[id] = torch.cuda.mem_get_info(id)[0]
        map_kwargs = {"max_memory": get_balanced_memory(model, max_memory)}
        device_map = infer_auto_device_map(model,
                                           no_split_module_classes=[
                                               "CustomLlamaDecoderLayer",
                                               "CustomGlmDecoderLayer",
                                               "CustomGLMBlock",
                                               "CustomQwen2DecoderLayer"
                                           ],
                                           **map_kwargs)
        model = dispatch_model(model, device_map=device_map)
        print(device_map)
    else:
        model = model.to(device_ids[0])

    generation_kwargs = {
        "build_dataset": True,
        "max_gpu_cache_memory": 0.0,
        "save_path": args.save_path,
        "buffer_size": 32768,
        "num_skip_layers": 2,
        "pos_sample_ratio": args.pos_sample_ratio,
    }
    generation_config = GenerationConfig(**generation_kwargs)

    longbench_v2_dataset = load_longbench_v2(args.longbench_v2_path)
    longbench_dataset = load_longbench(args.longbench_path)
    dataset = longbench_v2_dataset + longbench_dataset
    num_items = len(dataset)

    it = 0
    total_len = 0
    total_sample_len = 0
    for prompt in dataset:
        it += 1

        if args.apply_template:
            prompt = apply_template(prompt, model_name, tokenizer)
        encoded = tokenizer(prompt, return_tensors="pt")

        seq_len = encoded.input_ids.shape[1]
        if seq_len > args.max_context_length:
            input_ids = torch.cat(
                [
                    encoded.input_ids[:, :args.max_context_length // 2],
                    encoded.input_ids[:, -(args.max_context_length -
                                           args.max_context_length // 2):],
                ],
                dim=-1,
            ).to(model.device)
            attention_mask = torch.cat(
                [
                    encoded.attention_mask[:, :args.max_context_length // 2],
                    encoded.attention_mask[:,
                                           -(args.max_context_length -
                                             args.max_context_length // 2):],
                ],
                dim=-1,
            ).to(model.device)
        else:
            input_ids = encoded.input_ids.to(model.device)
            attention_mask = encoded.attention_mask.to(model.device)
        seq_len = input_ids.shape[1]

        # the idx of query to be sampled in this request
        query_idx = random.randint(seq_len // 2, seq_len)
        generation_config.query_idx = query_idx
        print(
            f"Sample {it}, sequence length {seq_len} sample {query_idx + 1} qks"
        )
        total_len += seq_len
        total_sample_len += query_idx + 1

        # set stop_mask = True for the last request
        # force the past_key_value to save not-fully-filled buffers at the last layer
        if it == num_items:
            generation_config.stop_mask = True
        else:
            generation_config.stop_mask = False

        input_len = input_ids.shape[1]
        output = model.generate(inputs=input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=1,
                                generation_config=generation_config)

    print(f"Total sequence length {total_len} sample {total_sample_len} qks")
