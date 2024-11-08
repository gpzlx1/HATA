import torch
import flashinfer


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
