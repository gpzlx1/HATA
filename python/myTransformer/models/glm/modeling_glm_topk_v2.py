import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import LayerNorm
from torch.nn.utils import skip_init
from typing import Optional, Tuple, Union, List
import transformers
from transformers.modeling_outputs import (
    BaseModelOutputWithPast, )
from transformers.utils import logging
from .modeling_chatglm import ChatGLMForConditionalGeneration, ChatGLMModel, GLMTransformer, GLMBlock, SelfAttention

from .configuration_chatglm import ChatGLMConfig

try:
    from transformers.utils import is_flash_attn_greater_or_equal_2_10
except:
    pass

from ..utils import SiLUAndMul
import flashinfer
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from ...cache.kvcache_topk import TopkStaticCache, prepare_cache_for_generation
import KVLib
import math

logger = logging.get_logger(__name__)


def _config_to_kwargs(args):
    common_kwargs = {
        "dtype": args.torch_dtype,
    }
    return common_kwargs


def default_init(cls, *args, **kwargs):
    return cls(*args, **kwargs)


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class CustomMLP(torch.nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config: ChatGLMConfig, device=None):
        super(CustomMLP, self).__init__()

        self.add_bias = config.add_bias_linear

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = nn.Linear(config.hidden_size,
                                       config.ffn_hidden_size * 2,
                                       bias=self.add_bias,
                                       device=device,
                                       **_config_to_kwargs(config))

        self.activation_func = SiLUAndMul()

        # Project back to h.
        self.dense_4h_to_h = nn.Linear(config.ffn_hidden_size,
                                       config.hidden_size,
                                       bias=self.add_bias,
                                       device=device,
                                       **_config_to_kwargs(config))

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


class CustomRotaryEmbedding(nn.Module):

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
            self.fn_kwargs['interleave'] = True
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
            self.fn_kwargs['interleave'] = True
            self.fn_kwargs['rope_scale'] = 1
            self.fn_kwargs['rope_theta'] = 10000.0 * config.rope_ratio
            self.fn = flashinfer.apply_rope

        self.rope_dim = int(config.kv_channels * 0.5)

    def forward(self, query_states, key_states, past_key_values):
        indptr, offsets = past_key_values.get_rope_metadata(
            query_states.device)
        fl_q, fl_k = self.fn(query_states,
                             key_states,
                             indptr,
                             offsets,
                             rotary_dim=self.rope_dim,
                             **self.fn_kwargs)
        return fl_q, fl_k


class CustomSelfAttention(SelfAttention):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        super(CustomSelfAttention, self).__init__(config, layer_number, device)
        self.rotary_emb = CustomRotaryEmbedding(config)
        self.is_causal = True
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10(
        )
        self.layer_idx = self.layer_number - 1
        self.scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[TopkStaticCache] = None,
        use_cache: bool = True,
    ):
        batch_size = past_key_value.curr_batch_size
        q_len = past_key_value.get_cur_q_len()
        is_prefill = q_len > 1

        mixed_x_layer = self.query_key_value(hidden_states)

        if self.multi_query_attention:
            (query_states, key_states, value_states) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition *
                    self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition *
                    self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition *
                    self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_states = query_states.view(query_states.size()[:-1] + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head))
            key_states = key_states.view(key_states.size()[:-1] + (
                self.num_multi_query_groups_per_partition,
                self.hidden_size_per_attention_head))
            value_states = value_states.view(value_states.size()[:-1] + (
                self.num_multi_query_groups_per_partition,
                self.hidden_size_per_attention_head))
        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                               (self.num_attention_heads_per_partition,
                                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [b, sq, np, 3 * hn] --> 3 [b, sq, np, hn]
            (query_states, key_states,
             value_states) = split_tensor_along_last_dim(mixed_x_layer, 3)

        query_states, key_states = self.rotary_emb(query_states, key_states,
                                                   past_key_value)

        query_states = query_states.view(
            batch_size, -1, self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head)
        key_states = key_states.view(batch_size, -1,
                                     self.num_multi_query_groups_per_partition,
                                     self.hidden_size_per_attention_head)
        value_states = value_states.view(
            batch_size, -1, self.num_multi_query_groups_per_partition,
            self.hidden_size_per_attention_head)

        torch.cuda.nvtx.range_push("append kcache")
        key_states = past_key_value.append(key_states,
                                           self.layer_idx,
                                           type="key",
                                           inc_seq_len=False)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("append vcache")
        value_states = past_key_value.append(value_states,
                                             self.layer_idx,
                                             type="value",
                                             inc_seq_len=True)
        torch.cuda.nvtx.range_pop()

        if is_prefill or self.layer_idx < past_key_value.get_num_skip_layers():
            attn_output = _flash_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                q_len,
                position_ids=position_ids,
                dropout=0,
                sliding_window=getattr(self, "sliding_window", None),
                use_top_left_mask=self._flash_attn_uses_top_left_mask,
                is_causal=self.is_causal,
            )
        else:
            torch.cuda.nvtx.range_push("compute topk")
            topk_indices = past_key_value.compute_topk(query_states,
                                                       self.layer_idx)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("sparse attention")
            attn_output, _ = KVLib.flash_index_decode(query_states, key_states,
                                                      value_states,
                                                      topk_indices, self.scale)
            torch.cuda.nvtx.range_pop()

        attn_output = attn_output.reshape(-1,
                                          self.projection_size).contiguous()
        output = self.dense(attn_output)

        return output, past_key_value


class CustomRMSNorm(torch.nn.Module):

    def __init__(self,
                 normalized_shape,
                 eps=1e-5,
                 device=None,
                 dtype=None,
                 **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(normalized_shape))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        output = flashinfer.norm.rmsnorm(hidden_states, self.weight,
                                         self.variance_epsilon)
        return output


class CustomGLMBlock(GLMBlock):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        super(CustomGLMBlock, self).__init__(config, layer_number, device)

        LayerNormFunc = CustomRMSNorm if config.rmsnorm else LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = LayerNormFunc(config.hidden_size,
                                             eps=config.layernorm_epsilon,
                                             device=device,
                                             dtype=config.torch_dtype)

        # Self attention.
        self.self_attention = CustomSelfAttention(config,
                                                  layer_number,
                                                  device=device)

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNormFunc(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            device=device,
            dtype=config.torch_dtype)
        # MLP
        self.mlp = CustomMLP(config, device=device)

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        kv_cache=None,
        use_cache=True,
    ):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, kv_cache = self.self_attention(
            layernorm_output,
            attention_mask,
            position_ids,
            past_key_value=kv_cache,
            use_cache=use_cache)

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = torch.nn.functional.dropout(attention_output,
                                                      p=self.hidden_dropout,
                                                      training=self.training)
        layernorm_input = residual + layernorm_input

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = torch.nn.functional.dropout(mlp_output,
                                             p=self.hidden_dropout,
                                             training=self.training)
        output = residual + output

        return output, kv_cache


class CustomGLMTransformer(GLMTransformer):
    """Transformer class."""

    def __init__(self, config: ChatGLMConfig, device=None):
        super(CustomGLMTransformer, self).__init__(config, device)

        # Transformer layers.
        def build_layer(layer_number):
            return CustomGLMBlock(config, layer_number, device=device)

        self.layers = torch.nn.ModuleList(
            [build_layer(i + 1) for i in range(self.num_layers)])

        if self.post_layer_norm:
            LayerNormFunc = CustomRMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = LayerNormFunc(config.hidden_size,
                                                 eps=config.layernorm_epsilon,
                                                 device=device,
                                                 dtype=config.torch_dtype)

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        kv_caches=None,
        use_cache: Optional[bool] = True,
        output_hidden_states: Optional[bool] = False,
    ):

        all_self_attentions = None
        all_hidden_states = () if output_hidden_states else None
        for index in range(self.num_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            layer = self._get_layer(index)
            layer_ret = layer(hidden_states,
                              attention_mask,
                              position_ids,
                              kv_cache=kv_caches,
                              use_cache=use_cache)
            hidden_states, kv_caches = layer_ret

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, kv_caches, all_hidden_states, all_self_attentions


class CustomChatGLMModel(ChatGLMModel):

    def __init__(self, config: ChatGLMConfig, device=None, empty_init=True):
        super().__init__(config)
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        init_kwargs = {}
        if device is not None:
            init_kwargs["device"] = device
        self.encoder = init_method(CustomGLMTransformer, config, **init_kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        full_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[TopkStaticCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if full_attention_mask is None:
            if (attention_mask is not None
                    and not attention_mask.all()) or (past_key_values
                                                      and seq_length != 1):
                full_attention_mask = self.get_masks(
                    input_ids, past_key_values, padding_mask=attention_mask)

        hidden_states = inputs_embeds
        bsz, seq_len, _ = hidden_states.shape

        # all the layers share the same allocation plan
        past_key_values.alloc(seq_length)

        inputs_embeds = inputs_embeds.view(bsz * seq_len, -1)
        # Run encoder.
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds,
            full_attention_mask,
            position_ids=position_ids,
            kv_caches=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states)
        hidden_states = hidden_states.view(bsz, seq_len, -1)
        if not return_dict:
            return tuple(v for v in [
                hidden_states, presents, all_hidden_states, all_self_attentions
            ] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class CustomGlmForCausalLM(ChatGLMForConditionalGeneration):

    def __init__(self, config: ChatGLMConfig, empty_init=True, device=None):
        super().__init__(config)
        self.transformer = CustomChatGLMModel(config,
                                              empty_init=empty_init,
                                              device=device)
        transformers.generation.utils.GenerationMixin._prepare_cache_for_generation = prepare_cache_for_generation
