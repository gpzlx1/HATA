from typing import Optional, Tuple, Union
import torch
import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

from ..cache import PagedCache, prepare_cache_for_generation
from .utils import register_flashinfer_attention

logger = logging.get_logger(__name__)


def customer_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[PagedCache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[
        torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states,
                                                    key_states,
                                                    cos,
                                                    sin,
                                                    unsqueeze_dim=2)

    kv_cache = past_key_value.update(key_states, value_states, self.layer_idx)
    query_states = query_states.view(bsz * q_len, self.num_heads,
                                     self.head_dim)

    attn_output = kwargs["attn_wrapper"].run(query_states, kv_cache)

    attn_output = attn_output.view(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def customer_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[PagedCache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[
        torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor,
                                             torch.FloatTensor]]]:

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


def customer_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[PagedCache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    register_flashinfer_attention(self, input_ids.device)

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (output_hidden_states if output_hidden_states
                            is not None else self.config.output_hidden_states)
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

    # all the layers share the same page allocation plan
    kv_indptr, kv_indices, kv_last_lens = past_key_values.alloc(bsz, seq_len)

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    kwargs = {}
    if seq_len == 1:
        self.decode_wrapper.plan(
            kv_indptr,
            kv_indices,
            kv_last_lens,
            self.layers[0].self_attn.num_heads,
            self.layers[0].self_attn.num_key_value_heads,
            self.layers[0].self_attn.head_dim,
            past_key_values.get_page_size(),
        )
        kwargs["attn_wrapper"] = self.decode_wrapper
    else:
        qo_indptr = torch.arange(0, (bsz + 1) * seq_len,
                                 seq_len,
                                 device=hidden_states.device,
                                 dtype=torch.int32)
        self.prefill_wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_lens,
            self.layers[0].self_attn.num_heads,
            self.layers[0].self_attn.num_key_value_heads,
            self.layers[0].self_attn.head_dim,
            past_key_values.get_page_size(),
            causal=True,
        )
        kwargs["attn_wrapper"] = self.prefill_wrapper

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states, )

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
                **kwargs,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1], )

    hidden_states = self.norm(hidden_states)

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


def enable():
    transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = customer_attention_forward
    transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = customer_decoder_layer_forward
    transformers.models.llama.modeling_llama.LlamaModel.forward = customer_model_forward
    transformers.generation.utils.GenerationMixin._prepare_cache_for_generation = prepare_cache_for_generation
