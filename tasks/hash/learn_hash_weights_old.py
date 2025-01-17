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

sys.path.append("../")

from dataloader import datasets_prompt
from llama_utils import get_model_type_arch

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model",
                        type=str,
                        default="/nfs/shared_LLM_model/THUDM/glm-4-9b-chat")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=
        "/nfs/shared_LLM_dataset/LongBench-v2/longbench-v2-top50-context-only.jsonl"
    )
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_context_length", type=int, default=131072)
    parser.add_argument("--save_path", type=str, default="./tmp")
    parser.add_argument("--cuda_mem", type=float, default=6.0)  # GB
    parser.add_argument("--rbit", type=int, default=128)
    parser.add_argument("--apply_template", default=False, action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=200)
    parser.add_argument("--train_epochs", type=int, default=1)
    parser.add_argument("--train_iters", type=int, default=640)
    parser.add_argument("--rep_iters", type=int, default=100)
    parser.add_argument("--sch_iters", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--pp_num", type=int, default=1)


# def load_processed_longbench_v2(path):
#     with open(path, "r", encoding="utf-8") as f:
#         lines = f.readlines()
#     return lines


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
    torch.manual_seed(42)

    os.makedirs(args.save_path, exist_ok=True)

    dataset = load_processed_longbench_v2(args.dataset_path)

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
        from myTransformer.models.glm.modeling_glm_fa_v2_learn_hash import CustomGlmForCausalLM
        model = CustomGlmForCausalLM.from_pretrained(args.model,
                                                     torch_dtype=torch.bfloat16,
                                                     config=config,
                                                     trust_remote_code=True)
    elif model_arch == "llama":
        from myTransformer.models.llama.modeling_llama_fa_learn_hash import CustomLlamaForCausalLM
        model = CustomLlamaForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, config=config)
        if model_name == "llama3":
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.padding_side = "left"
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
        "train": True,
        "max_gpu_cache_memory": args.cuda_mem * 1024 * 1024 * 1024,
        "hash_rbits": args.rbit,
        "save_path": args.save_path,
        "train_batch_size": args.train_batch_size,
        "train_epochs": args.train_epochs,
        "train_iters": args.train_iters,
        "rep_iters": args.rep_iters,
        "sch_iters": args.sch_iters,
        "lr": args.lr,
    }
    generation_config = GenerationConfig(**generation_kwargs)

    it = 0
    import random
    print(len(dataset))
    random.shuffle(dataset)
    dataset = dataset[:args.num_samples]
    for _ in range(5):
        random.shuffle(dataset)
        for prompt in dataset:
            it += 1
            # print(it)
            # continue
            # print(prompt)

            if args.apply_template:
                # llama3.1 & glm
                if model_name in ["llama3", "glm"]:
                    messages = [{"role": "user", "content": f"{prompt}"}]
                    prompt = tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False)
                # longchat
                elif model_name == "longchat":
                    conv = get_conversation_template("vicuna")
                    conv.append_message(conv.roles[0], prompt)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()

            encoded = tokenizer(prompt, return_tensors="pt")

            input_ids = encoded.input_ids.to(
                model.device)[:, :args.max_context_length]
            attention_mask = encoded.attention_mask.to(
                model.device)[:, :args.max_context_length]

            input_len = input_ids.shape[1]
            output = model.generate(inputs=input_ids,
                                    attention_mask=attention_mask,
                                    max_new_tokens=1,
                                    generation_config=generation_config)

            if it >= args.num_samples:
                break
