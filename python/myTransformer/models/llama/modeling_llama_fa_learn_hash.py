import torch
import torch.nn as nn
import transformers
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaModel,
    LlamaMLP,
    LlamaRMSNorm,
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

logger = logging.get_logger(__name__)

import os
import math

import time


class HashTrainer(torch.nn.Module):

    def __init__(
        self,
        num_kv_heads,
        num_heads,
        head_dim,
        rbit,
        save_path,
        layer_idx,
        batch_size=2000,
        schedule_iter=10,
        training_iter=50,
        report_iter=10,
        training_epoch=10,
        dtype=torch.bfloat16,
        device="cuda",
        lr=10,
    ):
        super(HashTrainer, self).__init__()
        self.rbit = rbit
        self.save_path = os.path.join(save_path,
                                      f"hash_weight_layer_{layer_idx:02d}.pt")
        self.layer_idx = layer_idx
        self.hash_weight = torch.randn((num_kv_heads, head_dim, self.rbit),
                                       requires_grad=True,
                                       dtype=dtype,
                                       device=device)
        self.optimizer = torch.optim.SGD([self.hash_weight],
                                         lr=lr,
                                         weight_decay=1e-6,
                                         momentum=0.9)
        # self.optimizer = torch.optim.Adam([self.hash_weight], lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                gamma=0.998)
        self.batch_size = batch_size
        self.schedule_iter = schedule_iter
        self.training_iter = training_iter
        self.report_iter = report_iter
        self.training_epoch = training_epoch
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.gqa_size = num_heads // num_kv_heads

    def loss_func(self,
                  q,
                  k,
                  q_hash,
                  k_hash,
                  epsilon_=1,
                  lambda_=1,
                  eta_=1,
                  sigma_=0.2):
        num_items = k.shape[2]
        q = q.reshape(self.num_heads, 1, self.head_dim)
        k = k.reshape(self.num_heads, num_items, self.head_dim)
        q_hash = q_hash.reshape(self.num_heads, 1, self.rbit)
        k_hash = k_hash.reshape(self.num_heads, num_items, self.rbit)
        q_hash = 2 * torch.sigmoid(sigma_ * q_hash) - 1
        k_hash = 2 * torch.sigmoid(sigma_ * k_hash) - 1

        eye = torch.eye(self.hash_weight.shape[1],
                        device=q.device,
                        requires_grad=False).unsqueeze(0)

        qk_similarity = (k @ q.transpose(-1, -2) /
                         math.sqrt(self.head_dim)).squeeze(
                             -1)  # (#heads, num_items)
        qk_similarity = torch.exp(qk_similarity - torch.max(
            qk_similarity, dim=-1, keepdim=True).values)  # (#heads, num_items)
        hash_similarity = torch.pow(torch.norm(q_hash - k_hash, dim=-1),
                                    2)  # (#heads, num_items)
        similarity_loss = epsilon_ * torch.sum(
            qk_similarity * hash_similarity) / math.sqrt(num_items)

        balance_loss = (q_hash + k_hash).sum(
            1, keepdim=True) / num_items / self.rbit  # (#heads, 1, rbit)
        balance_loss = lambda_ * (
            balance_loss @ balance_loss.transpose(-1, -2))
        balance_loss = balance_loss.sum()

        decorrelattion_loss = self.hash_weight.transpose(-1,
                                                         -2) @ self.hash_weight
        decorrelattion_loss = torch.norm(decorrelattion_loss -
                                         eye)  # / self.rbit
        decorrelattion_loss = eta_ * decorrelattion_loss

        loss = similarity_loss + balance_loss + decorrelattion_loss

        return loss, similarity_loss, balance_loss, decorrelattion_loss

    def train_for_one_epoch(self, query, key):
        query = query.to(self.dtype).detach()
        key = key.to(self.dtype).detach()

        input_bsz = query.shape[0]
        seq_len = key.shape[1]
        assert input_bsz == 1

        max_kiters = min(self.training_iter,
                         (seq_len + self.batch_size - 1) // self.batch_size)

        with torch.enable_grad():
            hash_weight = self.hash_weight.unsqueeze(1)
            for qiter in range(self.training_epoch):
                if qiter == 0:
                    curr_query = query[:, -(qiter + 1):, :, :].transpose(
                        1, 2).detach()
                else:
                    curr_query = query[:, -(qiter + 1):-qiter, :, :].transpose(
                        1, 2).detach()
                curr_query = curr_query.reshape(self.num_kv_heads,
                                                self.gqa_size, 1,
                                                self.head_dim)

                if qiter > 0:
                    valid_key = key[:, :-qiter, :, :]
                    valid_len = seq_len - qiter
                else:
                    valid_len = seq_len
                    valid_key = key
                rand_idx = torch.randperm(valid_len, device=key.device)
                valid_key = valid_key[:, rand_idx, :, :]

                for kiter in range(max_kiters):
                    batch_key = key[:, kiter * self.batch_size:(kiter + 1) *
                                    self.batch_size, :, :].transpose(
                                        1, 2).detach()

                    real_batch_size = batch_key.shape[2]
                    batch_key = batch_key.reshape(
                        self.num_kv_heads, 1, real_batch_size,
                        self.head_dim).expand(-1, self.gqa_size, -1, -1)

                    curr_query_code = curr_query @ hash_weight
                    batch_key_code = batch_key @ hash_weight
                    loss, similarity_loss, balance_loss, decorrelattion_loss = self.loss_func(
                        curr_query, batch_key, curr_query_code, batch_key_code)

                    self.optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.hash_weight,
                    #                                max_norm=1.0)
                    self.optimizer.step()
                    iter = qiter * max_kiters + kiter
                    # if (iter + 1) % self.schedule_iter == 0:
                    #     self.scheduler.step()
                    if iter % self.report_iter == 0:
                        # print(self.hash_weight)
                        print(
                            f"layer {self.layer_idx:2d} iter {iter:3d} sloss {similarity_loss.item():.5f} bloss {balance_loss.item():.5f} dloss {decorrelattion_loss.item():.5f} loss {loss.item():7.5f} "
                        )

    def save(self):
        torch.save(self.hash_weight.cpu(), self.save_path)


class CustomerLlamaMLP(LlamaMLP):

    def __init__(self, config):
        super().__init__(config)
        self.torch_dtype = config.torch_dtype
        self.mlp_bias = config.mlp_bias
        self.hidden_act = config.hidden_act
        assert self.hidden_act in ["silu"]

    def convert_fusion_exec(self):
        if not hasattr(self, "gate_up_proj"):
            device = self.down_proj.weight.device
            self.gate_up_proj = nn.Linear(self.hidden_size,
                                          self.intermediate_size * 2,
                                          bias=self.mlp_bias,
                                          dtype=self.torch_dtype,
                                          device=device)
            self.gate_up_proj.weight.data[:self.
                                          intermediate_size, :] = self.gate_proj.weight.data
            self.gate_up_proj.weight.data[
                self.intermediate_size:, :] = self.up_proj.weight.data
            self.act_fn = SiLUAndMul()

            del self.gate_proj
            del self.up_proj

    def forward(self, x):
        self.convert_fusion_exec()
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        x = self.down_proj(x)
        return x


class CustomLlamaRotaryEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config is not None
        if config.rope_scaling is not None:
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


class CustomLlamaAttention(LlamaFlashAttention2):

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.rotary_emb = CustomLlamaRotaryEmbedding(config)
        self.trainer = None

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

        if q_len > 1 and self.layer_idx > 1:
            if self.trainer is None:
                self.trainer = HashTrainer(
                    self.num_key_value_heads,
                    self.num_heads,
                    self.head_dim,
                    past_key_value.rbit,
                    past_key_value.save_path,
                    self.layer_idx,
                    dtype=torch.bfloat16,
                    device=query_states.device,
                    batch_size=past_key_value.train_batch_size,
                    training_epoch=past_key_value.train_epochs,
                    training_iter=past_key_value.train_iters,
                    report_iter=past_key_value.rep_iters,
                    schedule_iter=past_key_value.sch_iters,
                    lr=past_key_value.lr)

            # training here
            self.trainer.train_for_one_epoch(query_states, key_states)
            self.trainer.save()

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


class CustomLlamaRMSNorm(LlamaRMSNorm):

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        output = flashinfer.norm.rmsnorm(hidden_states, self.weight,
                                         self.variance_epsilon)
        return output


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
