import torch
import torch.nn as nn
import transformers
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaModel,
    LlamaDecoderLayer,
    LlamaFlashAttention2,
)
from transformers.utils import logging
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPast

from ...cache.kvcache_fa_for_training import CustomStaticCacheForTraining, prepare_cache_for_generation
from ..utils import SiLUAndMul
import flashinfer
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from ...cache.kernels.triton_qk_score import qk_score

from .llama_utils import CustomerLlamaMLP, CustomLlamaRMSNorm, CustomLlamaRotaryEmbedding

logger = logging.get_logger(__name__)


class CustomLlamaAttention(LlamaFlashAttention2):

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.rotary_emb = CustomLlamaRotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[CustomStaticCacheForTraining] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor,
                  torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:

        batch_size = past_key_value.curr_batch_size
        q_len = past_key_value.get_cur_q_len()
        _, hidden_size = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(-1, self.num_heads, self.head_dim)

        key_states = self.k_proj(hidden_states)
        key_states = key_states.view(-1, self.num_key_value_heads,
                                     self.head_dim)

        value_states = self.v_proj(hidden_states)
        value_states = value_states.view(-1, self.num_key_value_heads,
                                         self.head_dim)

        query_states, key_states = self.rotary_emb(query_states, key_states,
                                                   past_key_value)

        query_states = query_states.view(batch_size, -1, self.num_heads,
                                         self.head_dim)
        key_states = key_states.view(batch_size, -1, self.num_key_value_heads,
                                     self.head_dim)
        value_states = value_states.view(batch_size, -1,
                                         self.num_key_value_heads,
                                         self.head_dim)

        if q_len > 1 and self.layer_idx >= past_key_value.num_skip_layers:
            query_idx = min(past_key_value.query_idx, q_len)
            assert batch_size == 1, "batch size must be 1 when building dataset"
            select_query = query_states[:, query_idx:query_idx + 1, :, :]
            select_key = key_states[:, :query_idx + 1, :, :]
            select_qk_score = qk_score(select_query, select_key,
                                       query_idx + 1).squeeze(0)  # (h, s)

            # select_qk_score = torch.exp(
            #     select_qk_score -
            #     torch.max(select_qk_score, dim=-1, keepdim=True).values)

            topk_indices = torch.topk(select_qk_score,
                                      dim=-1,
                                      k=int(select_qk_score.shape[-1] *
                                            past_key_value.pos_sample_ratio),
                                      largest=True).indices
            select_qk_score[:, :] = -1.0
            topk_scores = torch.linspace(
                20.0,
                1.0,
                steps=topk_indices.shape[-1],
                dtype=select_qk_score.dtype,
                device=select_qk_score.device).unsqueeze(0).expand(
                    self.num_heads, -1)
            select_qk_score = torch.scatter(select_qk_score,
                                            dim=-1,
                                            index=topk_indices,
                                            src=topk_scores)
            past_key_value.save_data(select_query.squeeze(0),
                                     select_key.squeeze(0), select_qk_score,
                                     self.layer_idx)

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
        attn_output = attn_output.view(-1, hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class CustomLlamaDecoderLayer(LlamaDecoderLayer):

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.self_attn = CustomLlamaAttention(config, layer_idx)
        self.input_layernorm = CustomLlamaRMSNorm(config.hidden_size,
                                                  eps=config.rms_norm_eps)
        self.post_attention_layernorm = CustomLlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = CustomerLlamaMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[CustomStaticCacheForTraining] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor,
                  torch.Tensor]] = None,  # will become mandatory in v4.46
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
            cache_position=cache_position,
            position_embeddings=position_embeddings,
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


class CustomLlamaModel(LlamaModel):

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([
            CustomLlamaDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = CustomLlamaRMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.rotary_emb = None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[CustomStaticCacheForTraining] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length(
            ) if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens,
                                          past_seen_tokens +
                                          inputs_embeds.shape[1],
                                          device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds,
                                               cache_position, past_key_values,
                                               output_attentions)
        hidden_states = inputs_embeds
        bsz, seq_len, _ = hidden_states.shape

        # all the layers share the same allocation plan
        past_key_values.alloc(seq_len)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        kwargs = {}

        hidden_states = hidden_states.view(bsz * seq_len, -1)
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states, )

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=None,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[
                    2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1], )

        if past_key_values.stop_mask:
            past_key_values.force_save_buffers()

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(bsz, seq_len, -1)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states, )

        next_cache = next_decoder_cache if use_cache else None

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


class CustomLlamaForCausalLM(LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.model = CustomLlamaModel(config)
        transformers.generation.utils.GenerationMixin._prepare_cache_for_generation = prepare_cache_for_generation
