from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import argparse
import torch
import os
import numpy as np
import random
from modeling_llama_infinigen_preprocess import replace_llama_infinigen_preprocess


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate partial weight")
    parser.add_argument(
        "--model",
        default="/nfs/shared_LLM_model/meta-llama/Llama-2-7b-chat-hf",
        help='model')
    parser.add_argument("--prompt_file",
                        default="pg19_firstbook.txt",
                        help='model')
    parser.add_argument("--output",
                        default="preprocessed",
                        help='output directory to store result')
    args = parser.parse_args()
    seed_everything(42)

    replace_llama_infinigen_preprocess()
    config = AutoConfig.from_pretrained(args.model)
    config._attn_implementation = "flash_attention_2"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 torch_dtype=torch.float16,
                                                 config=config).eval().cuda()

    model_name = os.path.basename(os.path.normpath(args.model))
    head_dim = model.model.layers[0].self_attn.head_dim
    n_head = model.model.layers[0].self_attn.num_heads
    n_kv_head = model.model.layers[0].self_attn.num_key_value_heads
    gqa_size = n_head // n_kv_head
    n_layer = config.num_hidden_layers

    print("Generate skewing...")

    for i in range(n_layer):
        model.model.layers[i].self_attn.config.gen_partial = False

    with open(args.prompt_file, 'r') as file:
        prompt = file.read()
    input_ids = tokenizer(prompt,
                          return_tensors="pt").input_ids.cuda()[:, :2048]
    generated_ids = model.generate(input_ids,
                                   max_new_tokens=1,
                                   min_new_tokens=1)

    print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
    skewing_matrix = torch.zeros((n_layer, n_head, head_dim, head_dim),
                                 device="cuda",
                                 dtype=torch.float16)
    for l, layer in enumerate(model.model.layers):
        query = layer.self_attn.rope_query
        key = layer.self_attn.rope_key

        for head in range(n_head):
            in_q = query[0, head]
            in_k = key[0, head // gqa_size]
            uq, sq, vq = torch.svd(in_q.to(torch.float))
            uk, sk, vk = torch.svd(in_k.to(torch.float))
            s = sq * sk
            a = torch.zeros(head_dim, head_dim).to('cuda')
            _, ind = s.sort()
            r, c = a.shape
            skewing_matrix[l, head] = a.scatter(-1,
                                                ind.unsqueeze(0).repeat(r, 1),
                                                vq).to(torch.float16)

    skewing_matrix = skewing_matrix.cpu()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    for l, layer in enumerate(model.model.layers):
        torch.save(skewing_matrix[l],
                   os.path.join(args.output, f"skewing_martix_{l:02d}.pt"))
