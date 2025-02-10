import torch
from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig
import time
import numpy as np
import myTransformer
import json
from datasets import Dataset
from fastchat.model import get_conversation_template
from transformers.generation.configuration_utils import GenerationConfig
import random

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# dataset_prompt = 'Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: '
# dataset_path = "/nfs/shared_LLM_dataset/LongBench/data/passage_retrieval_en_e.jsonl"

dataset_prompt = 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:'
dataset_path = "/nfs/shared_LLM_dataset/LongBench/data/qasper_e.jsonl"

with open(dataset_path, "r") as f:
    lines = f.readlines()

dataset = []
for line in lines:
    eg = json.loads(line)
    instance = {
        "context": eg["context"],
        "input": eg["input"],
        "answers": [eg["answers"]]
    }
    instance["length"] = len(instance["context"].split())
    instance["all_classes"] = None
    dataset.append(instance)

dataset = Dataset.from_list(dataset)

if __name__ == "__main__":
    # i = int(sys.argv[1])
    device = "cuda:0"
    torch.cuda.set_device(device)
    torch.manual_seed(42)

    # model_path = "/nfs/shared_LLM_model/meta-llama/Llama-2-7b-chat-hf"
    # model_path = "/nfs/shared_LLM_model/lmsys/longchat-7b-v1.5-32k"
    # model_path = "/nfs/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct"
    # model_path = "/nfs/shared_LLM_model/THUDM/glm-4-9b-chat"
    model_path = "/nfs/shared_LLM_model/Qwen/Qwen2.5-14B-Instruct-1M"
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)

    from myTransformer.models.llama.modeling_llama_multi_hash import CustomLlamaForCausalLM
    from myTransformer.models.glm.modeling_glm_multi_hash_v2 import CustomGlmForCausalLM
    from myTransformer.models.qwen2.modeling_qwen2_multi_hash import CustomQwen2ForCausalLM

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.torch_dtype = torch.float16
    config._attn_implementation = "flash_attention_2"
    print(config)

    if "glm" in model_path.lower():
        model = CustomGlmForCausalLM.from_pretrained(model_path,
                                                     torch_dtype=torch.float16,
                                                     config=config,
                                                     trust_remote_code=True)
    elif "qwen" in model_path.lower():
        model = CustomQwen2ForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, config=config)
    else:
        model = CustomLlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, config=config)
    # print(model)

    model = model.eval().to(device)

    generation_kwargs = {
        "max_gpu_cache_memory": 10 * 1024 * 1024 * 1024,  # 30GB
        "page_num": 1000,
        "page_size": 16,
        "hash_rbits": 128,
        "hash_weights_path":
        # "/root/workspace/myoffloading/model_weights_v5/Meta-Llama-3.1-8B-Instruct-128",
        None,
        "sparse_ratio": 512,
        "use_norm": False,
        "num_sink": 0,
        "num_recent": 0,
    }
    generation_config = GenerationConfig(**generation_kwargs)

    it = 0

    for ctx in dataset:

        prompt = dataset_prompt.format(context=ctx["context"],
                                       input=ctx["input"])

        # llama3.1 & glm
        if any([
                x in model_path.lower() for x in [
                    "llama-3.1", "llama3.1", "llama_3.1", "llama-3", "llama3",
                    "llama_3", "glm"
                ]
        ]):
            messages = [{"role": "user", "content": f"{prompt}"}]
            prompt = tokenizer.apply_chat_template(messages,
                                                   add_generation_prompt=True,
                                                   tokenize=False)
        # longchat
        elif any([x in model_path.lower() for x in ["longchat"]]):
            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
        elif any([x in model_path.lower() for x in ["qwen"]]):
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(messages,
                                                   add_generation_prompt=True,
                                                   tokenize=False)

        encoded = tokenizer(prompt, return_tensors="pt")

        input_ids = encoded.input_ids.to(model.device)
        attention_mask = encoded.attention_mask.to(model.device)

        input_len = input_ids.shape[1]
        output = model.generate(inputs=input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=32,
                                generation_config=generation_config)

        output = tokenizer.decode(output[0][input_len:],
                                  skip_special_tokens=True)
        print()
        print(output)

        it += 1
        if it > 2:
            break
