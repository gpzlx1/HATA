from flash_attn import flash_attn_with_kvcache
from .llama_utils import (
    CustomLlamaRotaryEmbedding,
    CustomerLlamaMLP,
    CustomLlamaRMSNorm,
)
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

from ...cache.kvcache_fa import CustomStaticCache, prepare_cache_for_generation


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
        past_key_value: Optional[CustomStaticCache] = None,
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

        key_states = past_key_value.append(key_states,
                                           self.layer_idx,
                                           type="key",
                                           inc_seq_len=False)
        value_states = past_key_value.append(value_states,
                                             self.layer_idx,
                                             type="value",
                                             inc_seq_len=True)

        query_states = query_states.view(batch_size, -1, self.num_heads,
                                         self.head_dim)

        attn_output = flash_attn_with_kvcache(
            query_states,
            key_states,
            value_states,
            causal=True,
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
        past_key_value: Optional[CustomStaticCache] = None,
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
        past_key_values: Optional[CustomStaticCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        assert inputs_embeds is None, "inputs_embeds is not supported in CustomLlamaModel"
        output_attentions = False
        # output_hidden_states = False
        use_cache = True
        # return_dict = True

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            raise ValueError(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )

        # chunk prefill here
        CHUNK_SIZE = 4096
        for chunk_start in range(0, input_ids.shape[1], CHUNK_SIZE):
            chunk_input_ids = input_ids[:,
                                        chunk_start:chunk_start + CHUNK_SIZE]

            chunk_inputs_embeds = self.embed_tokens(chunk_input_ids)

            hidden_states = chunk_inputs_embeds
            bsz, q_len, _ = hidden_states.shape

            # all the layers share the same allocation plan
            past_key_values.alloc(q_len)

            # decoder layers
            all_hidden_states = None
            all_self_attns = None
            next_decoder_cache = None

            kwargs = {}

            hidden_states = hidden_states.view(bsz * q_len, -1)

            for decoder_layer in self.layers:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=None,
                    position_ids=None,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=None,
                    position_embeddings=None,
                    **kwargs,
                )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = layer_outputs[
                        2 if output_attentions else 1]

        # get last hidden state
        hidden_states = hidden_states.view(
            bsz, q_len, -1)[:, -1, :].view(bsz, -1)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(bsz, 1, -1)

        next_cache = next_decoder_cache

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
