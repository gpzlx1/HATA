import sys
import tqdm
import torch
import os
import argparse
from transformers import AutoConfig, AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig
from functools import partial
from sklearn.decomposition import PCA
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

sys.path.append("../")
from dataloader import RULERManager
from llama_utils import get_model_type_arch


def set_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model",
        type=str,
        default="/nfs/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/nfs/shared_LLM_dataset/RULER/Meta-Llama-3.1-8B-Instruct/128K"
    )
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--key_type", type=str, default="pre_rotary")
    parser.add_argument("--max_context_length", type=int, default=32768)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pp_num", type=int, default=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_args(parser)
    args = parser.parse_args()

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
        from modeling_glm_fa_save_key import CustomGlmForCausalLM
        model = CustomGlmForCausalLM.from_pretrained(args.model,
                                                     torch_dtype=torch.float16,
                                                     config=model_config,
                                                     trust_remote_code=True)
    elif model_arch == "llama":
        from modeling_llama_fa_save_key import CustomLlamaForCausalLM
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

    dataset_manager = RULERManager(
        args.dataset_path,
        args.dataset_path,
    )
    task = dataset_manager.get_dataset_names()[-2]
    print(f"Get key tensors on {task} dataset from RULER")
    _, dataset_maxlen, _ = dataset_manager.get_dataset_info(task)
    raw_data = dataset_manager.get_data(task)

    process_fn = partial(dataset_manager.process_raw_data,
                         tokenizer=tokenizer,
                         apply_chat_template=None,
                         task=None)
    remove_columns = []
    for key in raw_data[0]:
        if key not in ["length", "all_classes", "answers", "depth_percent"]:
            remove_columns.append(key)
    encoded_data = raw_data.map(
        process_fn,
        batched=True,
        num_proc=4,
        batch_size=10,
        with_indices=True,
        remove_columns=remove_columns,
    )

    generation_kwargs = {
        "max_gpu_cache_memory": 0.0 * 1024 * 1024 * 1024,
    }
    generation_config = GenerationConfig(**generation_kwargs)

    num_layers = model_config.num_hidden_layers
    if model_arch == "glm":
        num_heads = model_config.multi_query_group_num
    else:
        num_heads = (model_config.num_attention_heads if getattr(
            model_config, "num_key_value_heads", None) is None else
                     model_config.num_key_value_heads)
    head_dim = (model_config.head_dim if hasattr(model_config, "head_dim") else
                model_config.hidden_size // model_config.num_attention_heads)
    key_tensors = [[] for _ in range(num_layers)]

    cnt = 0
    seq_len = args.max_context_length
    for i in tqdm.tqdm(range(args.num_samples),
                       desc="get prefilled key tensors"):
        cnt += 1
        input = {
            "input_ids":
            torch.Tensor([encoded_data["input_ids"][i]]).long().cuda(),
            "attention_mask":
            torch.Tensor([encoded_data["attention_mask"][i]]).long().cuda()
        }

        model.generate(**input,
                       do_sample=False,
                       max_new_tokens=dataset_maxlen,
                       generation_config=generation_config)
        for l in range(num_layers):
            if model_arch == "glm":
                if args.key_type == "pre_rotary":
                    seq_len = min(
                        model.transformer.encoder.layers[l].self_attention.
                        pre_rotary_key.shape[0], seq_len)
                    key_tensors[l].append(model.transformer.encoder.layers[l].
                                          self_attention.pre_rotary_key.cpu())
                else:
                    seq_len = min(
                        model.transformer.encoder.layers[l].self_attention.
                        post_rotary_key.shape[0], seq_len)
                    key_tensors[l].append(model.transformer.encoder.layers[l].
                                          self_attention.post_rotary_key.cpu())
            else:
                if args.key_type == "pre_rotary":
                    seq_len = min(
                        model.model.layers[l].self_attn.pre_rotary_key.
                        shape[0], seq_len)
                    key_tensors[l].append(
                        model.model.layers[l].self_attn.pre_rotary_key.cpu())
                else:
                    seq_len = min(
                        model.model.layers[l].self_attn.post_rotary_key.
                        shape[0], seq_len)
                    key_tensors[l].append(
                        model.model.layers[l].self_attn.post_rotary_key.cpu())

    del model

    os.makedirs(os.path.join(args.output_dir, f"pca_components"),
                exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, f"pca_means"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, f"pca_explained_variance"),
                exist_ok=True)

    pca_components = torch.zeros(num_heads, head_dim, head_dim)
    pca_means = torch.zeros(num_heads, head_dim)
    pca_explained_variance = torch.zeros(num_heads, head_dim)
    for l in tqdm.tqdm(range(num_layers), desc="Run pca on layers"):
        for i in range(args.num_samples):
            key_tensors[l][i] = key_tensors[l][i][:seq_len, :, :]
        layer_keys = torch.stack(key_tensors[l])
        for h in range(num_heads):
            head_keys = layer_keys[:, :, h, :].view(-1, layer_keys.shape[-1])
            pca = PCA()
            pca.fit(head_keys.numpy())
            pca_components[h] = torch.tensor(pca.components_)
            pca_means[h] = torch.tensor(pca.mean_)
            pca_explained_variance[h] = torch.tensor(
                pca.explained_variance_ratio_)

            torch.save(
                pca_components,
                os.path.join(
                    args.output_dir,
                    f"pca_components/pca_components_layer_{l:02d}.pt"))
            torch.save(
                pca_components,
                os.path.join(args.output_dir,
                             f"pca_means/pca_means_layer_{l:02d}.pt"))
            torch.save(
                pca_components,
                os.path.join(
                    args.output_dir,
                    f"pca_explained_variance/pca_explained_variance_layer_{l:02d}.pt"
                ))
