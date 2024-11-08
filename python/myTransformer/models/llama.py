from typing import Optional, Tuple
import torch
import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
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
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:

    register_flashinfer_attention(self, hidden_states.device)

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

    kv_cache, kv_indptr, kv_indices, kv_last_lens = past_key_value.update(
        key_states, value_states, self.layer_idx)
    query_states = query_states.view(bsz * q_len, self.num_heads,
                                     self.head_dim)

    if q_len > 1:
        qo_indptr = torch.arange(0, (bsz + 1) * q_len,
                                 q_len,
                                 device=query_states.device,
                                 dtype=torch.int32)
        self.prefill_wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_lens,
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim,
            past_key_value.get_page_size(),
            causal=True,
        )
        attn_output = self.prefill_wrapper.run(query_states, kv_cache)
    else:
        self.decode_wrapper.plan(
            kv_indptr,
            kv_indices,
            kv_last_lens,
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim,
            past_key_value.get_page_size(),
        )
        attn_output = self.decode_wrapper.run(query_states, kv_cache)

    attn_output = attn_output.view(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def enable():
    transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = customer_attention_forward
    transformers.generation.utils.GenerationMixin._prepare_cache_for_generation = prepare_cache_for_generation
