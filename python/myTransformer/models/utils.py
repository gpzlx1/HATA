import torch
import torch.nn as nn
import flashinfer
from flashinfer.activation import silu_and_mul
from flash_attn import flash_attn_func
from typing import List, Tuple


def flash_attnention(q, k, v, scale):
    attn, lse, _ = flash_attn_func(q,
                                   k,
                                   v,
                                   softmax_scale=scale,
                                   return_attn_probs=True)
    return attn, lse


def combine_attention(attns: List[torch.Tensor] | Tuple[torch.Tensor],
                      lses: List[torch.Tensor] | Tuple[torch.Tensor]):
    batch_size = attns[0].shape[0]
    num_heads = attns[0].shape[2]
    dtype = attns[0].dtype

    lses = [lse.view(batch_size, 1, num_heads, 1) for lse in lses]

    lses = torch.cat(lses, dim=1)
    lse_logsum = torch.exp(lses).sum(dim=1, keepdim=True)

    attns = torch.cat(attns, dim=1)

    attn = (torch.exp(lses) * attns).sum(dim=1, keepdim=True) / lse_logsum

    return attn.to(dtype)


def register_flashinfer_attention(attn, device):
    if hasattr(attn, "prefill_wrapper"):
        return
    prefill_workspace = torch.empty(128 * 1024 * 1024,
                                    dtype=torch.uint8,
                                    device=device)
    decode_workspace = torch.empty(128 * 1024 * 1024,
                                   dtype=torch.uint8,
                                   device=device)
    attn.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        prefill_workspace, "NHD")
    attn.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        decode_workspace, "NHD")


class SiLUAndMul(nn.Module):

    def __init__(self):
        super(SiLUAndMul, self).__init__()

    def forward(self, x):
        return silu_and_mul(x)
