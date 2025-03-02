import torch
import torch.nn as nn
import transformers
from .modeling_deepseek import (
    DeepseekV2ForCausalLM,
    DeepseekV2Model,
    DeepseekV2MoE,
    DeepseekV2MLP,
    DeepseekV2RMSNorm,
    DeepseekV2DecoderLayer,
    DeepseekV2FlashAttention2,
    AddAuxiliaryLoss,
    rotate_half,
)

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask, )
from transformers.modeling_outputs import (
    BaseModelOutputWithPast, )

from transformers.pytorch_utils import (
    is_torch_greater_or_equal_than_1_13, )
from transformers.utils import (
    logging, )
from transformers.utils.import_utils import is_torch_fx_available
from .configuration_deepseek import DeepseekV2Config

import flashinfer
from ...cache.kvcache_mla import CustomStaticCache, prepare_cache_for_generation
from ..utils import SiLUAndMul
import torch.distributed as dist
import numpy as np

# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(
        _prepare_4d_causal_attention_mask)

logger = logging.get_logger(__name__)


class CustomDeepseekV2RMSNorm(DeepseekV2RMSNorm):

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps)

    def forward(self, hidden_states):
        b, slen, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        output = flashinfer.norm.rmsnorm(hidden_states, self.weight,
                                         self.variance_epsilon)
        output = output.view(b, slen, hidden_size)
        return output


class CustomDeepseekV2MLP(DeepseekV2MLP):

    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__(config, hidden_size, intermediate_size)
        self.torch_dtype = config.torch_dtype
        self.hidden_act = config.hidden_act
        self.converted = False
        assert self.hidden_act in ["silu"]

    def convert_fusion_exec(self):
        if not hasattr(self, "gate_up_proj"):
            device = self.down_proj.weight.device
            self.gate_up_proj = nn.Linear(self.hidden_size,
                                          self.intermediate_size * 2,
                                          bias=False,
                                          dtype=self.torch_dtype,
                                          device=device)

            self.gate_up_proj.weight.data[:self.
                                          intermediate_size, :] = self.gate_proj.weight.data
            del self.gate_proj

            self.gate_up_proj.weight.data[
                self.intermediate_size:, :] = self.up_proj.weight.data.to(
                    device)
            del self.up_proj

            self.act_fn = SiLUAndMul()

            torch.cuda.empty_cache()

    def forward(self, x):
        self.convert_fusion_exec()
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        x = self.down_proj(x)
        return x

    # def forward(self, x):
    #     down_proj = self.down_proj(
    #         self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    #     return down_proj


class CustomDeepseekV2MoE(DeepseekV2MoE):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        if hasattr(config, "ep_size") and config.ep_size > 1:
            self.experts = nn.ModuleList([
                (CustomDeepseekV2MLP(
                    config, intermediate_size=config.moe_intermediate_size)
                 if i >= self.ep_rank * self.experts_per_rank
                 and i < (self.ep_rank + 1) * self.experts_per_rank else None)
                for i in range(config.n_routed_experts)
            ])
        else:
            self.ep_size = 1
            self.experts_per_rank = config.n_routed_experts
            self.ep_rank = 0
            self.experts = nn.ModuleList([
                CustomDeepseekV2MLP(
                    config, intermediate_size=config.moe_intermediate_size)
                for i in range(config.n_routed_experts)
            ])
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = CustomDeepseekV2MLP(
                config=config, intermediate_size=intermediate_size)

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(
                self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(
                    hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) *
                 topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.to(hidden_states.dtype).view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, topk_idx,
                               topk_weight).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x: torch.Tensor, topk_ids: torch.Tensor,
                  topk_weight: torch.Tensor):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)  # (#tokens, #experts)
        tokens_per_expert = cnts.sum(dim=0)  # (#experts)
        idxs = topk_ids.view(-1).argsort()

        tokens_per_expert = tokens_per_expert.cpu().numpy()
        outs = torch.empty((idxs.shape[0], x.shape[1]),
                           dtype=x.dtype,
                           device=x.device)

        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            outs[idxs[start_idx:end_idx]] = expert(x[idxs[start_idx:end_idx] //
                                                     topk_ids.shape[1]])
            start_idx = end_idx

        final_out = (outs.view(
            *topk_ids.shape, -1).type(topk_weight.dtype).mul_(
                topk_weight.unsqueeze(dim=-1)).sum(dim=1).type(outs.dtype))
        return final_out


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    b, s, h, d = q.shape
    b, s, hkv, d = k.shape
    # cos/sin: (seqlen, rope_dim)

    cos = cos[position_ids].view(1, s, 1, d)
    sin = sin[position_ids].view(1, s, 1, d)

    q = q.view(b, s, h, d // 2, 2).transpose(4, 3).reshape(b, s, h, d)
    k = k.view(b, s, hkv, d // 2, 2).transpose(4, 3).reshape(b, s, hkv, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2 with Llama->DeepseekV2
class CustomDeepseekV2Attention(DeepseekV2FlashAttention2):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[CustomStaticCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim)
        cos, sin = self.rotary_emb(k_pe,
                                   seq_len=past_key_value.get_seq_length() +
                                   q_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        compressed_kv = past_key_value.append(compressed_kv,
                                              self.layer_idx,
                                              type="lora",
                                              inc_seq_len=False)
        kv_seq_len = compressed_kv.shape[1]
        k_pe = past_key_value.append(k_pe,
                                     self.layer_idx,
                                     type="key",
                                     inc_seq_len=True)
        k_pe = k_pe.view(bsz, kv_seq_len, 1, self.qk_rope_head_dim)

        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv)).view(
            bsz, kv_seq_len, self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim)
        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        query_states = k_pe.new_empty(bsz, q_len, self.num_heads,
                                      self.q_head_dim)
        query_states[:, :, :, :self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim:] = q_pe

        key_states = k_pe.new_empty(bsz, kv_seq_len, self.num_heads,
                                    self.q_head_dim)
        key_states[:, :, :, :self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim:] = k_pe

        value_states = F.pad(value_states,
                             [0, self.q_head_dim - self.v_head_dim])

        dropout_rate = self.attention_dropout if self.training else 0.0

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            softmax_scale=self.softmax_scale,
        )
        if self.q_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, :self.v_head_dim]

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads *
                                          self.v_head_dim).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class CustomDeepseekV2DecoderLayer(DeepseekV2DecoderLayer):

    def __init__(self, config: DeepseekV2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = CustomDeepseekV2Attention(config=config,
                                                   layer_idx=layer_idx)
        self.mlp = (CustomDeepseekV2MoE(config) if
                    (config.n_routed_experts is not None
                     and layer_idx >= config.first_k_dense_replace
                     and layer_idx % config.moe_layer_freq == 0) else
                    CustomDeepseekV2MLP(config))
        self.input_layernorm = CustomDeepseekV2RMSNorm(config.hidden_size,
                                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = CustomDeepseekV2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[CustomStaticCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor,
                                                 torch.FloatTensor]]]:
        if hidden_states.device.index != torch.cuda.current_device():
            torch.cuda.set_device(hidden_states.device)

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, )

        if output_attentions:
            outputs += (self_attn_weights, )

        if use_cache:
            outputs += (present_key_value, )

        return outputs


class CustomDeepseekV2Model(DeepseekV2Model):

    def __init__(self, config: DeepseekV2Config):
        super().__init__(config)
        self.layers = nn.ModuleList([
            CustomDeepseekV2DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = CustomDeepseekV2RMSNorm(config.hidden_size,
                                            eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[CustomStaticCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (output_attentions if output_attentions is not None
                             else self.config.output_attentions)
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (return_dict if return_dict is not None else
                       self.config.use_return_dict)

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
                )
                use_cache = False

        past_key_values_length = past_key_values.get_seq_length(
        ) if past_key_values is not None else 0

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = (attention_mask if
                              (attention_mask is not None
                               and 0 in attention_mask) else None)
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        # embed positions
        hidden_states = inputs_embeds
        bsz, seq_len, _ = hidden_states.shape
        past_key_values.alloc(seq_len)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states, )

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[
                    2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1], )

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states, )

        next_cache = next_decoder_cache
        if not return_dict:
            return tuple(
                v for v in
                [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class CustomDeepseekV2ForCausalLM(DeepseekV2ForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.model = CustomDeepseekV2Model(config)
        transformers.generation.utils.GenerationMixin._prepare_cache_for_generation = prepare_cache_for_generation

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values: Optional[CustomStaticCache] = None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.get_seq_length()
            max_cache_length = past_key_values.get_max_length()

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (attention_mask is not None
                    and attention_mask.shape[1] > input_ids.shape[1]):
                input_ids = input_ids[:, -(attention_mask.shape[1] -
                                           past_length):]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (max_cache_length is not None and attention_mask is not None
                    and cache_length + input_ids.shape[1] > max_cache_length):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        return model_inputs
