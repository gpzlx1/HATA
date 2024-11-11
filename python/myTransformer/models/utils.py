import torch
import torch.nn as nn
import flashinfer
from flashinfer.activation import silu_and_mul


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
