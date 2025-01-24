import torch
import numpy as np
import myTransformer
import argparse
import time
import math
from flash_attn import flash_attn_func


def set_args(parser: argparse.ArgumentParser):
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--num_kv_heads", type=int, default=32)
    parser.add_argument("--rbit", type=int, default=128)
    parser.add_argument("--seqlens", type=str, default="32000,64000,96000")
    parser.add_argument("--num_decode_steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sparse_budget", type=int, default=512)


def torch_decode_hash_encode(key: torch.Tensor, hash_weight: torch.Tensor):
    b, _, h, d = key.shape
    hkv, _, r = hash_weight.shape
    gqa = h // hkv

    if gqa > 1:
        key = key.view(b, hkv, gqa, d)
    else:
        key = key.view(b, h, 1, d)
    hash_weight = hash_weight.unsqueeze(0)

    key_code = torch.matmul(key, hash_weight) > 0
    key_code = key_code.view(b, 1, h, r)

    return key_code


@torch.compile()
def torch_hamming_distance_kernel(query_code: torch.Tensor,
                                  key_code: torch.Tensor,
                                  key_norm: torch.Tensor, rbit: int):
    hamming_distance = (key_code ^ query_code).sum(dim=-1).transpose(-1, -2)
    hash_score = (1 - 2 * hamming_distance.to(key_norm.dtype) /
                  rbit) * key_norm.transpose(-1, -2)
    return hash_score.contiguous()


def torch_hamming_distance(query_code: torch.Tensor,
                           key_code_cache: torch.Tensor,
                           key_norm_cache: torch.Tensor, seq_len: int):
    b, _, hkv, r = key_code_cache.shape
    h = query_code.shape[2]
    gqa = h // hkv

    if gqa > 1:
        key_code = key_code_cache[:, :seq_len, :, :].unsqueeze(-2).expand(
            -1, -1, -1, gqa, -1).reshape(b, seq_len, h, r)
        key_norm = key_norm_cache[:, :seq_len, :].unsqueeze(-1).expand(
            -1, -1, -1, gqa).reshape(b, seq_len, h)
    else:
        key_code = key_code_cache[:, :seq_len, :, :]
        key_norm = key_norm_cache[:, :seq_len, :]

    hash_score = torch_hamming_distance_kernel(query_code, key_code, key_norm,
                                               r)

    if gqa > 1:
        hash_score = hash_score.view(b, hkv, gqa, seq_len).sum(2)
    return hash_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_args(parser)
    args = parser.parse_args()
    print(args)
    torch.cuda.set_device(args.device)

    seqlens = args.seqlens.split(",")
    seqlens = [int(seqlen) for seqlen in seqlens]
    dtype = torch.float16
    scale = 1 / math.sqrt(args.head_dim)
    hash_weights = torch.randn((args.num_kv_heads, args.head_dim, args.rbit),
                               dtype=dtype,
                               device=args.device)

    for seqlen in seqlens:
        key_cache = torch.randn(
            (args.batch_size, seqlen + args.num_decode_steps + args.warmup,
             args.num_kv_heads, args.head_dim),
            dtype=dtype,
            device=args.device)
        value_cache = torch.randn(
            (args.batch_size, seqlen + args.num_decode_steps + args.warmup,
             args.num_kv_heads, args.head_dim),
            dtype=dtype,
            device=args.device)
        key_code_cache = torch.randint(
            0,
            1, (args.batch_size, seqlen + args.num_decode_steps + args.warmup,
                args.num_kv_heads, args.rbit),
            dtype=bool,
            device=args.device)
        key_norm_cache = torch.randn(
            (args.batch_size, seqlen + args.num_decode_steps + args.warmup,
             args.num_kv_heads),
            dtype=dtype,
            device=args.device)

        curr_seq_len = seqlen
        total_steps = args.num_decode_steps + args.warmup
        time_log = []
        for i in range(total_steps):
            query = torch.randn(
                (args.batch_size, 1, args.num_heads, args.head_dim),
                dtype=dtype,
                device=args.device)
            key = torch.randn(
                (args.batch_size, 1, args.num_kv_heads, args.head_dim),
                dtype=dtype,
                device=args.device)
            value = torch.randn(
                (args.batch_size, 1, args.num_kv_heads, args.head_dim),
                dtype=dtype,
                device=args.device)
            key_cache[:, curr_seq_len:curr_seq_len + 1, :, :] = key
            value_cache[:, curr_seq_len:curr_seq_len + 1, :, :] = value

            tic = time.time()
            torch.cuda.synchronize()
            # ======================== hash encode ===========================
            query_code = torch_decode_hash_encode(query, hash_weights)
            key_code = torch_decode_hash_encode(key, hash_weights)
            key_norm = torch.norm(key, dim=-1)
            key_code_cache[:, curr_seq_len:curr_seq_len + 1, :, :] = key_code
            key_norm_cache[:, curr_seq_len:curr_seq_len + 1, :] = key_norm
            curr_seq_len += 1
            # ===================== hash score & topk ========================
            hash_score = torch_hamming_distance(query_code, key_code_cache,
                                                key_norm_cache, curr_seq_len)
            topk_index = myTransformer.capi.batch_topk(hash_score,
                                                       args.sparse_budget,
                                                       True)
            # ===================== gather & attention =======================
            topk_index = topk_index.transpose(-1, -2).unsqueeze(-1).expand(
                args.batch_size, args.sparse_budget, args.num_kv_heads,
                args.head_dim).to(torch.int64)
            sparse_key = torch.gather(key_cache, dim=1, index=topk_index)
            sparse_value = torch.gather(value_cache, dim=1, index=topk_index)
            flash_attn_func(query,
                            sparse_key,
                            sparse_value,
                            softmax_scale=scale,
                            return_attn_probs=False)
            # ================================================================
            torch.cuda.synchronize()
            toc = time.time()
            time_log.append(toc - tic)

        decode_time = np.mean(time_log[args.warmup:]) * 1000 * 1000
        print(
            f"Prefill len {seqlen:6d} | Decode steps {total_steps:4d} | Decode latency {decode_time:10.3f} μs"
        )
