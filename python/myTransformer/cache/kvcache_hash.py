from typing import Dict, Optional, Union, Any
import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.configuration_utils import GenerationConfig
from .kernels.triton_hash_encode import prefill_hash_encode, decode_hash_encode
from .kvcache_fa import CustomStaticCache
import KVLib
import os


class HashStaticCache(CustomStaticCache):

    def __init__(
        self,
        config: PretrainedConfig,
        hash_rbits: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
        max_gpu_cache_memory_size: int = 1000000000,  # 0.93 GB
        layer_device_map: Optional[Dict[int, Union[str, torch.device,
                                                   int]]] = None,
        sparse_ratio: float = 0.1,
        num_skip_layers: int = 2,
        hash_weights_path: str = None,
        use_norm: bool = False,
        num_sink: int = 0,
        num_recent: int = 0,
        max_batch_size: int = 16,
        reuse_cos_threshold=0.8,
        reuse_topk=False,
    ) -> None:
        super().__init__(config, device, dtype, max_gpu_cache_memory_size,
                         layer_device_map)
        self.hash_rbits = hash_rbits
        self.sparse_ratio = sparse_ratio
        self.hash_weights_path = hash_weights_path
        self.num_skip_layers = num_skip_layers
        self.use_norm = use_norm

        self.num_sink = num_sink
        self.num_recent = num_recent
        self.max_batch_size = max_batch_size
        self.max_gpu_cache_memory_size -= 2 * self.num_layers * self.max_batch_size * (
            self.num_sink + self.num_recent
        ) * self.num_key_value_heads * self.head_dim * self.dtype.itemsize

        self.gqa_size = self.num_heads // self.num_key_value_heads

        self.reuse_cos_threshold = reuse_cos_threshold
        self.reuse_topk = reuse_topk

        self.hash_packbit_aux_tensors = {}

    def build_cache(self):

        self.layer_caches = []
        self.max_layer_caches = []

        self.layer_hash_caches = []
        self.max_layer_hash_caches = []

        self.layer_norm_caches = []
        self.max_layer_norm_caches = []

        self.layer_sink_recent_caches = []

        self.hash_weights = []

        assert self.hash_rbits % 32 == 0
        self.hash_dim = self.hash_rbits // 32

        per_token_per_head_kv_size = self.dtype.itemsize * self.head_dim * 2
        per_token_per_head_hash_size = torch.int32.itemsize * self.hash_dim + self.dtype.itemsize

        all_layer_per_token_per_head_kv_size = per_token_per_head_kv_size * self.num_layers
        all_layer_per_token_per_head_hash_size = per_token_per_head_hash_size * (
            self.num_layers - self.num_skip_layers)
        all_layer_per_token_per_head_size = all_layer_per_token_per_head_kv_size + all_layer_per_token_per_head_hash_size

        self.max_kv_cache_size = self.max_gpu_cache_memory_size * all_layer_per_token_per_head_kv_size / all_layer_per_token_per_head_size
        self.max_hash_cache_size = self.max_gpu_cache_memory_size * all_layer_per_token_per_head_hash_size / all_layer_per_token_per_head_size
        self.max_norm_cache_size = self.max_hash_cache_size * self.dtype.itemsize / per_token_per_head_hash_size
        self.max_hash_cache_size -= self.max_norm_cache_size

        self.each_layer_max_kv_cache = self.max_kv_cache_size / self.num_layers
        self.each_layer_max_hash_cache = self.max_hash_cache_size / (
            self.num_layers - self.num_skip_layers)
        self.each_layer_max_norm_cache = self.max_norm_cache_size / (
            self.num_layers - self.num_skip_layers)

        kv_numel = int(self.each_layer_max_kv_cache / self.dtype.itemsize)
        self.each_layer_max_kv_numel = kv_numel
        hash_numel = int(self.each_layer_max_hash_cache / torch.int32.itemsize)
        self.each_layer_max_hash_numel = hash_numel
        norm_numel = int(self.each_layer_max_norm_cache / self.dtype.itemsize)
        self.each_layer_max_norm_numel = norm_numel

        for l in range(self.num_layers):
            layer_device = self.layer_devices[l]

            self.layer_caches.append(None)
            self.layer_hash_caches.append(None)
            self.layer_norm_caches.append(None)

            self.max_layer_caches.append(
                torch.zeros((kv_numel, ),
                            dtype=self.dtype,
                            device=layer_device))

            self.layer_sink_recent_caches.append(None)

            if l >= self.num_skip_layers:
                self.max_layer_hash_caches.append(
                    torch.zeros((hash_numel, ),
                                dtype=torch.int32,
                                device=layer_device))
                self.max_layer_norm_caches.append(
                    torch.zeros((norm_numel, ),
                                dtype=self.dtype,
                                device=layer_device))
                if self.hash_weights_path is None:
                    hash_weight = torch.randn((self.head_dim, self.hash_rbits),
                                              dtype=self.dtype,
                                              device=layer_device)
                else:
                    hash_weight = torch.load(
                        os.path.join(
                            self.hash_weights_path,
                            f"hash_weight_layer_{l:02d}.pt")).to(layer_device)
                self.hash_weights.append(hash_weight)
            else:
                self.max_layer_hash_caches.append(None)
                self.max_layer_norm_caches.append(None)
                self.hash_weights.append(None)

        self.max_seq_len = 0
        self.curr_batch_size = 0
        self.seq_len = 0
        self.layer_hash_seq_lens = [0 for l in range(self.num_layers)]
        self.layer_cache_lens = [0 for _ in range(self.num_layers)]
        self.layer_recent_insert_ptr = [0 for _ in range(self.num_layers)]

        for device in self.unique_devices:
            self.hash_packbit_aux_tensors[device] = torch.pow(
                2, torch.arange(0, 32, 1, dtype=torch.int32, device=device))

        self.query_code_buffers = None

    def reset(self, batch_size):
        self.curr_batch_size = batch_size
        self.seq_len = 0
        self.layer_cache_lens = [0 for _ in range(self.num_layers)]
        self.layer_hash_seq_lens = [0 for _ in range(self.num_layers)]
        self.layer_recent_insert_ptr = [0 for _ in range(self.num_layers)]

        # shape for KVCache: (batch_size, 2, max_seq_len, num_heads * head_dim)
        kv_max_seq_len = self.each_layer_max_kv_numel // (
            self.num_key_value_heads * self.head_dim * self.curr_batch_size *
            2)
        hash_max_seq_len = self.each_layer_max_hash_numel // (
            self.num_key_value_heads * self.hash_dim * self.curr_batch_size)
        norm_max_seq_len = self.each_layer_max_norm_numel // (
            self.num_key_value_heads * self.curr_batch_size)
        self.max_seq_len = min(kv_max_seq_len, hash_max_seq_len,
                               norm_max_seq_len)

        # print(self.max_seq_len)

        numel = 2 * batch_size * self.max_seq_len * self.num_key_value_heads * self.head_dim
        hash_numel = batch_size * self.max_seq_len * self.num_key_value_heads * self.hash_dim
        norm_numel = batch_size * self.max_seq_len * self.num_key_value_heads

        for i in range(self.num_layers):
            self.layer_caches[i] = self.max_layer_caches[i][:numel]
            self.layer_caches[i] = self.layer_caches[i].view(
                2, batch_size, self.max_seq_len, self.num_key_value_heads,
                self.head_dim)

            self.layer_sink_recent_caches[i] = torch.zeros(
                (2, batch_size, self.num_sink + self.num_recent,
                 self.num_key_value_heads, self.head_dim),
                device=self.layer_devices[i],
                dtype=self.dtype)

            if i >= self.num_skip_layers:
                self.layer_hash_caches[i] = self.max_layer_hash_caches[
                    i][:hash_numel]
                self.layer_hash_caches[i] = self.layer_hash_caches[i].view(
                    batch_size, self.max_seq_len, self.num_key_value_heads,
                    self.hash_dim)

                self.layer_norm_caches[i] = self.max_layer_norm_caches[
                    i][:norm_numel]
                self.layer_norm_caches[i] = self.layer_norm_caches[i].view(
                    batch_size, self.max_seq_len, self.num_key_value_heads)

        self.query_code_buffers = {}
        for device in self.unique_devices:
            self.query_code_buffers[device] = torch.empty(batch_size,
                                                          1,
                                                          self.num_heads,
                                                          self.hash_dim,
                                                          dtype=torch.int32,
                                                          device=device)

        self.prev_query = None
        self.curr_query = None

        self.layer_cached_query = [None for _ in range(self.num_layers)]
        self.layer_cached_topk = [None for _ in range(self.num_layers)]
        self.layer_reuse_mask = [None for _ in range(self.num_layers)]

        self.prefill_len = 0

    def append_prefill(self, key_states: torch.Tensor,
                       value_states: torch.Tensor, layer_idx: int):
        self.prefill_len = key_states.shape[1]
        prefill_len = key_states.shape[1]

        if layer_idx == self.num_layers - 1:
            self.seq_len += prefill_len

        # sink
        self.layer_sink_recent_caches[layer_idx][
            0, :, :self.num_sink, :, :] = key_states[:, :self.num_sink, :, :]
        self.layer_sink_recent_caches[layer_idx][
            1, :, :self.num_sink, :, :] = value_states[:, :self.num_sink, :, :]

        # recent
        self.layer_sink_recent_caches[layer_idx][
            0, :, self.num_sink:, :, :] = key_states[:,
                                                     -self.num_recent:, :, :]
        self.layer_sink_recent_caches[layer_idx][
            1, :, self.num_sink:, :, :] = value_states[:,
                                                       -self.num_recent:, :, :]
        self.layer_recent_insert_ptr[layer_idx] = self.num_sink

        residual_num = prefill_len - self.num_sink - self.num_recent

        # middle
        self.layer_caches[layer_idx][
            0, :, :residual_num, :, :] = key_states[:, self.num_sink:-self.
                                                    num_recent, :, :]
        self.layer_caches[layer_idx][
            1, :, :residual_num, :, :] = value_states[:, self.num_sink:-self.
                                                      num_recent, :, :]

        self.layer_cache_lens[layer_idx] += residual_num

    def append_decode(self, key_states: torch.Tensor,
                      value_states: torch.Tensor, layer_idx: int):

        KVLib.kvcache_append(self.layer_sink_recent_caches[layer_idx],
                             key_states, value_states,
                             self.layer_recent_insert_ptr[layer_idx])

        self.layer_recent_insert_ptr[layer_idx] += 1
        if self.layer_recent_insert_ptr[
                layer_idx] == self.num_sink + self.num_recent:
            self.layer_recent_insert_ptr[
                layer_idx] = self.num_sink  # circular queue

        if layer_idx == self.num_layers - 1:
            self.seq_len += 1

    def value_proj_and_append(self,
                              hidden_states,
                              value_weightsT,
                              layer_idx,
                              inc_seq_len=False):

        hidden_size = value_weightsT.shape[0]
        kv_hidden_size = value_weightsT.shape[1]
        hidden_states = hidden_states.view(self.curr_batch_size, -1,
                                           hidden_size)

        insert_cache = self.layer_sink_recent_caches[layer_idx].view(
            2, self.curr_batch_size, -1, kv_hidden_size)
        insert_ptr = self.layer_recent_insert_ptr[layer_idx]
        torch.matmul(hidden_states,
                     value_weightsT,
                     out=insert_cache[1, :, insert_ptr:insert_ptr + 1, :])

        if inc_seq_len:
            self.layer_recent_insert_ptr[layer_idx] += 1
            if self.layer_recent_insert_ptr[
                    layer_idx] == self.num_sink + self.num_recent:
                self.layer_recent_insert_ptr[
                    layer_idx] = self.num_sink  # circular queue

            if layer_idx == len(self.layer_caches) - 1:
                self.seq_len += 1

    def prefill_encode_hash(self, layer_idx):
        assert layer_idx >= self.num_skip_layers, f"hash topk is not enabled in layer{layer_idx}!"
        key = self.layer_caches[layer_idx][
            0, :, :self.layer_cache_lens[layer_idx], :, :]
        prefill_hash_encode(
            key, self.hash_weights[layer_idx],
            self.layer_hash_caches[layer_idx],
            self.layer_norm_caches[layer_idx],
            self.hash_packbit_aux_tensors[self.layer_devices[layer_idx]])
        self.layer_hash_seq_lens[layer_idx] = self.layer_cache_lens[layer_idx]

    def decode_encode_hash(self, query, layer_idx):
        assert layer_idx >= self.num_skip_layers, f"hash topk is not enabled in layer{layer_idx}!"

        ptr = self.layer_recent_insert_ptr[layer_idx]
        key = self.layer_sink_recent_caches[layer_idx][0, :, ptr:ptr + 1, :, :]

        # KVLib.decode_hash_encode(key, self.hash_weights[layer_idx],
        decode_hash_encode(
            key, self.hash_weights[layer_idx],
            self.layer_hash_caches[layer_idx],
            self.layer_norm_caches[layer_idx], query,
            self.query_code_buffers[self.layer_devices[layer_idx]],
            self.hash_packbit_aux_tensors[self.layer_devices[layer_idx]],
            self.layer_hash_seq_lens[layer_idx])
        self.layer_hash_seq_lens[layer_idx] += 1

        return self.query_code_buffers[self.layer_devices[layer_idx]]

    def advance_recent_window(self, layer_idx):
        cache_len = self.layer_cache_lens[layer_idx]
        ptr = self.layer_recent_insert_ptr[layer_idx]
        KVLib.kvcache_append2(self.layer_caches[layer_idx],
                              self.layer_sink_recent_caches[layer_idx],
                              cache_len, ptr)
        self.layer_cache_lens[layer_idx] += 1

    def get_middle_cache(self, layer_idx, truncate=False):
        if truncate:
            key = self.layer_caches[layer_idx][
                0, :, :self.layer_cache_lens[layer_idx], :, :]
            value = self.layer_caches[layer_idx][
                1, :, :self.layer_cache_lens[layer_idx], :, :]
        else:
            key = self.layer_caches[layer_idx][0]
            value = self.layer_caches[layer_idx][1]
        return key, value, self.layer_cache_lens[layer_idx]

    def get_sink_recent_cache(self, layer_idx):
        return self.layer_sink_recent_caches[layer_idx][
            0], self.layer_sink_recent_caches[layer_idx][1]

    def compute_topk(self, encoded_query: torch.Tensor, layer_idx: int):
        assert layer_idx >= self.num_skip_layers, f"hash topk is not enabled in layer{layer_idx}!"
        torch.cuda.nvtx.range_push("hash score")
        score = KVLib.hamming_score(self.layer_hash_caches[layer_idx],
                                    encoded_query,
                                    self.layer_norm_caches[layer_idx],
                                    self.hash_rbits,
                                    self.layer_hash_seq_lens[layer_idx],
                                    self.use_norm)
        largest = True if self.use_norm else False
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("compute topk")
        if self.sparse_ratio < 1:
            fetch_num = min(
                int(self.seq_len * self.sparse_ratio) - self.num_recent -
                self.num_sink, self.layer_hash_seq_lens[layer_idx])
        else:
            fetch_num = min(
                int(self.sparse_ratio) - self.num_recent - self.num_sink,
                self.layer_hash_seq_lens[layer_idx])
        topk_indices = KVLib.batch_topk(score, fetch_num, largest)
        # topk_indices = torch.topk(score, fetch_num, dim=-1,
        #                           largest=largest).indices.int()
        torch.cuda.nvtx.range_pop()

        if self.reuse_topk:
            if self.layer_reuse_mask[layer_idx] is not None:
                topk_indices = torch.where(self.layer_reuse_mask[layer_idx],
                                           self.layer_cached_topk,
                                           topk_indices)
            self.layer_cached_topk = topk_indices

        return topk_indices

    def get_num_skip_layers(self):
        return self.num_skip_layers

    def register_query(self, query):
        self.prev_query = self.curr_query
        self.curr_query = query

    def get_query(self):
        return self.prev_query

    def update_registered_query(self, query):
        self.prev_query = query

    def check_reuse(self, query, layer_idx):
        if self.reuse_topk:
            if self.layer_cached_query[layer_idx] is not None:
                cos_sim = torch.cosine_similarity(
                    self.layer_cached_query[layer_idx], query, dim=-1)
                cos_sim = cos_sim.view(self.curr_batch_size,
                                       self.num_key_value_heads,
                                       self.gqa_size).mean(dim=-1,
                                                           keepdim=True)
                self.layer_reuse_mask[
                    layer_idx] = cos_sim > self.reuse_cos_threshold
                expand_reuse_mask = self.layer_reuse_mask[layer_idx].expand(
                    self.curr_batch_size, self.num_key_value_heads,
                    self.gqa_size).reshape(self.curr_batch_size, 1,
                                           self.num_heads, 1)
                self.layer_cached_query[layer_idx] = torch.where(
                    expand_reuse_mask, self.layer_cached_query[layer_idx],
                    query)

            else:
                self.layer_cached_query[layer_idx] = query


"""
===================================================
Hugging Face api reload
===================================================
"""


def prepare_cache_for_generation(
    self,
    generation_config: GenerationConfig,
    model_kwargs: Dict,
    assistant_model,
    batch_size: int,
    max_cache_length: int,
    device: torch.device,
) -> bool:
    if not hasattr(self, "_cache"):

        def get_layer_device_map(execution_device_map: Optional[dict] = None):
            if execution_device_map is None or len(execution_device_map) <= 1:
                return None
            layer_device_map = {}
            for layer in execution_device_map:
                for idx in range(self.config.num_hidden_layers):
                    if f".{idx}." in f"{layer}.":
                        layer_device_map[idx] = execution_device_map[layer]
                        break
            for idx in range(self.config.num_hidden_layers):
                if idx not in layer_device_map:
                    raise RuntimeError(
                        f"layer {idx} has not been mapped to a device.")
            return layer_device_map

        execution_device_map = None
        if hasattr(self, "hf_device_map"):
            main_device = [
                d for d in self.hf_device_map.values()
                if d not in ["cpu", "disk"]
            ][0]
            execution_device_map = {
                name: main_device if device in ["cpu", "disk"] else device
                for name, device in self.hf_device_map.items()
            }

        layer_device_map = get_layer_device_map(execution_device_map)
        self._cache = HashStaticCache(
            config=self.config.get_text_config(),
            hash_rbits=generation_config.hash_rbits,
            max_gpu_cache_memory_size=generation_config.max_gpu_cache_memory,
            device=device,
            dtype=self.dtype,
            layer_device_map=layer_device_map,
            sparse_ratio=generation_config.sparse_ratio,
            hash_weights_path=generation_config.hash_weights_path,
            use_norm=generation_config.use_norm,
            num_sink=generation_config.num_sink,
            num_recent=generation_config.num_recent,
        )
        self._cache.build_cache()

    self._cache.reset(batch_size)
    cache_name = "past_key_values"
    model_kwargs[cache_name] = self._cache
