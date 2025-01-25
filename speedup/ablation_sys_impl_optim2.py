import torch
import numpy as np
import myTransformer
import argparse
import time
import math


def set_args(parser: argparse.ArgumentParser):
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--num_kv_heads", type=int, default=32)
    parser.add_argument("--rbit", type=int, default=128)
    parser.add_argument("--seqlens", type=str, default="96000,128000")
    parser.add_argument("--num_decode_steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sparse_budget", type=float, default=0.016)


def torch_decode_hash_encode(key: torch.Tensor, hash_weight: torch.Tensor,
                             pack_bits_tensor: torch.Tensor):
    b, _, h, d = key.shape
    hkv, _, r = hash_weight.shape
    gqa = h // hkv

    if gqa > 1:
        key = key.view(b, hkv, gqa, d)
    else:
        key = key.view(b, h, 1, d)
    hash_weight = hash_weight.unsqueeze(0)

    key_code = torch.matmul(key, hash_weight) > 0
    key_code = key_code.view(b, 1, h, r // 32, 32)
    key_code = (key_code.to(pack_bits_tensor.dtype) *
                pack_bits_tensor).sum(dim=-1)

    return key_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_args(parser)
    args = parser.parse_args()
    print(args)
    torch.cuda.set_device(args.device)

    seqlens = args.seqlens.split(",")
    seqlens = [int(seqlen) for seqlen in seqlens]
    dtype = torch.float16
    hash_dim = args.rbit // 32
    scale = 1 / math.sqrt(args.head_dim)
    hash_weights = torch.randn((args.num_kv_heads, args.head_dim, args.rbit),
                               dtype=dtype,
                               device=args.device)
    hash_packbit_aux_tensors = torch.pow(
        2, torch.arange(0, 32, 1, dtype=torch.int32, device=args.device))

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
            torch.iinfo(torch.int32).min,
            torch.iinfo(torch.int32).max,
            (args.batch_size, seqlen + args.num_decode_steps + args.warmup,
             args.num_kv_heads, hash_dim),
            dtype=torch.int32,
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
            query_code = torch_decode_hash_encode(query, hash_weights,
                                                  hash_packbit_aux_tensors)
            key_code = torch_decode_hash_encode(key, hash_weights,
                                                hash_packbit_aux_tensors)
            key_norm = torch.norm(key, dim=-1)
            key_code_cache[:, curr_seq_len:curr_seq_len + 1, :, :] = key_code
            key_norm_cache[:, curr_seq_len:curr_seq_len + 1, :] = key_norm
            curr_seq_len += 1
            # ===================== hash score & topk ========================
            hash_score = myTransformer.capi.hamming_score(key_code_cache,
                                                          query_code,
                                                          key_norm_cache,
                                                          args.rbit,
                                                          curr_seq_len,
                                                          use_key_norm=True)
            sparse_budget = int(
                args.sparse_budget) if args.sparse_budget >= 1 else int(
                    curr_seq_len * args.sparse_budget)
            topk_index = myTransformer.capi.batch_topk(hash_score,
                                                       sparse_budget, True)
            # ===================== gather & attention =======================
            myTransformer.capi.flash_index_decode(query, key_cache,
                                                  value_cache, topk_index,
                                                  scale)
            # ================================================================
            torch.cuda.synchronize()
            toc = time.time()
            time_log.append(toc - tic)

        decode_time = np.mean(time_log[args.warmup:]) * 1000 * 1000
        print(
            f"Prefill len {seqlen:6d} | Decode steps {total_steps:4d} | Decode latency {decode_time:10.3f} μs"
        )
