from ..utils import SiLUAndMul
import flashinfer
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2MLP,
    Qwen2RMSNorm,
)
import torch.nn as nn
import torch


class CustomerQwen2MLP(Qwen2MLP):

    def __init__(self, config):
        super().__init__(config)
        self.torch_dtype = config.torch_dtype
        self.hidden_act = config.hidden_act
        self.converted = False
        assert self.hidden_act in ["silu"]

    def convert_fusion_exec(self):
        if not self.converted:
            device = self.down_proj.weight.device
            self.gate_up_proj = nn.Linear(self.hidden_size,
                                          self.intermediate_size * 2,
                                          bias=False,
                                          dtype=self.torch_dtype,
                                          device=device)
            self.gate_up_proj.weight.data[:self.
                                          intermediate_size, :] = self.gate_proj.weight.data
            self.gate_up_proj.weight.data[
                self.intermediate_size:, :] = self.up_proj.weight.data
            self.act_fn = SiLUAndMul()

            del self.gate_proj
            del self.up_proj
            self.converted = True

    def forward(self, x):
        self.convert_fusion_exec()
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        x = self.down_proj(x)
        return x


class CustomQwen2RotaryEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config is not None
        if "rope_scaling" in config and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"

        assert self.rope_type in ["default", "llama3", "linear"]

        self.fn = None
        self.fn_kwargs = {}

        if self.rope_type == "linear":
            self.fn_kwargs['interleave'] = False
            self.fn_kwargs['rope_scale'] = config.rope_scaling["factor"]
            self.fn_kwargs['rope_theta'] = config.rope_theta
            self.fn = flashinfer.apply_rope

        elif self.rope_type == "llama3":
            self.fn_kwargs['interleave'] = False
            self.fn_kwargs['high_freq_factor'] = config.rope_scaling[
                'high_freq_factor']
            self.fn_kwargs['low_freq_factor'] = config.rope_scaling[
                'low_freq_factor']
            self.fn_kwargs['rope_theta'] = config.rope_theta
            self.fn_kwargs['rope_scale'] = config.rope_scaling['factor']
            self.fn_kwargs['old_context_len'] = config.rope_scaling[
                'original_max_position_embeddings']
            self.fn = flashinfer.apply_llama31_rope

        elif self.rope_type == "default":
            self.fn_kwargs['interleave'] = False
            self.fn_kwargs['rope_scale'] = 1
            self.fn_kwargs['rope_theta'] = config.rope_theta
            self.fn = flashinfer.apply_rope

    def forward(self, query_states, key_states, past_key_values):
        indptr, offsets = past_key_values.get_rope_metadata(
            query_states.device)
        fl_q, fl_k = self.fn(query_states, key_states, indptr, offsets,
                             **self.fn_kwargs)
        return fl_q, fl_k


class CustomQwen2RMSNorm(Qwen2RMSNorm):

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        output = flashinfer.norm.rmsnorm(hidden_states, self.weight,
                                         self.variance_epsilon)
        return output
