from typing import Dict, Optional, Union, Any
import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.configuration_utils import GenerationConfig
from .kvcache_fa import CustomStaticCache
import os


def qk_score(query, key, seq_len):
    b, _, hk, d = key.shape
    h = query.shape[-2]

    query = query.transpose(1, 2).transpose(-1, -2)
    key = key[:, :seq_len, :, :].transpose(1, 2).unsqueeze(2).expand(
        -1, -1, h // hk, -1, -1).reshape(b, h, seq_len, d)
    score = key @ query
    return score.squeeze(-1)


class InfiniGenStaticCache(CustomStaticCache):

    def __init__(
        self,
        config: PretrainedConfig,
        num_channels: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
        max_gpu_cache_memory_size: int = 1000000000,  # 0.93 GB
        layer_device_map: Optional[Dict[int, Union[str, torch.device,
                                                   int]]] = None,
        sparse_ratio: float = 0.1,
        num_skip_layers: int = 2,
        skewing_matrix_path: str = None,
    ) -> None:
        super().__init__(config, device, dtype, max_gpu_cache_memory_size,
                         layer_device_map)
        self.partial_dim = num_channels
        self.sparse_ratio = sparse_ratio
        self.skewing_matrix_path = skewing_matrix_path
        self.num_skip_layers = num_skip_layers
        self.gqa_size = 1
        self.num_key_value_heads = self.num_heads

    def build_cache(self):

        self.layer_caches = []
        self.max_layer_caches = []

        self.layer_partial_caches = []
        self.max_layer_partial_caches = []

        self.skewing_matrix = []

        kv_ratio = 2 * self.num_layers * self.head_dim
        partial_key_ratio = (self.num_layers -
                             self.num_skip_layers) * self.partial_dim

        self.max_kv_cache_size = self.max_gpu_cache_memory_size * kv_ratio / (
            kv_ratio + partial_key_ratio)
        self.max_partial_cache_size = self.max_gpu_cache_memory_size * partial_key_ratio / (
            kv_ratio + partial_key_ratio)
        each_layer_max_kv_cache = self.max_kv_cache_size / self.num_layers
        each_layer_max_partial_cache = self.max_partial_cache_size / (
            self.num_layers - self.num_skip_layers)

        kv_numel = int(each_layer_max_kv_cache / self.dtype.itemsize)
        self.each_layer_max_kv_numel = kv_numel
        partial_numel = int(each_layer_max_partial_cache / self.dtype.itemsize)
        self.each_layer_max_partial_numel = partial_numel

        for l in range(self.num_layers):
            layer_device = self.layer_devices[l]

            self.layer_caches.append(None)
            self.layer_partial_caches.append(None)

            self.max_layer_caches.append(
                torch.zeros((kv_numel, ),
                            dtype=self.dtype,
                            device=layer_device))

            if l >= self.num_skip_layers:
                self.max_layer_partial_caches.append(
                    torch.zeros((partial_numel, ),
                                dtype=self.dtype,
                                device=layer_device))
                self.skewing_matrix.append(
                    torch.load(
                        os.path.join(self.skewing_matrix_path,
                                     f"skewing_martix_{l:02d}.pt")).to(
                                         layer_device)[None, :, :, :])
            else:
                self.max_layer_partial_caches.append(None)
                self.skewing_matrix.append(None)

        self.max_seq_len = 0
        self.curr_batch_size = 0
        self.seq_len = 0
        self.layer_partial_seq_lens = [0 for l in range(self.num_layers)]

        self.partial_idx = [None for _ in range(self.num_layers)]

    def reset(self, batch_size):
        self.curr_batch_size = batch_size
        self.seq_len = 0
        self.layer_partial_seq_lens = [0 for l in range(self.num_layers)]

        kv_max_seq_len = self.each_layer_max_kv_numel // (
            self.num_key_value_heads * self.head_dim * self.curr_batch_size *
            2)
        partial_max_seq_len = self.each_layer_max_partial_numel // (
            self.num_key_value_heads * self.partial_dim * self.curr_batch_size)
        self.max_seq_len = min(kv_max_seq_len, partial_max_seq_len)

        numel = 2 * batch_size * self.max_seq_len * self.num_key_value_heads * self.head_dim
        partial_numel = batch_size * self.max_seq_len * self.num_key_value_heads * self.partial_dim

        for i in range(self.num_layers):
            self.layer_caches[i] = self.max_layer_caches[i][:numel]
            self.layer_caches[i] = self.layer_caches[i].view(
                2, batch_size, self.max_seq_len, self.num_key_value_heads,
                self.head_dim)

            if i >= self.num_skip_layers:
                self.layer_partial_caches[i] = self.max_layer_partial_caches[
                    i][:partial_numel]
                self.layer_partial_caches[i] = self.layer_partial_caches[
                    i].view(batch_size, self.max_seq_len,
                            self.num_key_value_heads, self.partial_dim)

        self.partial_idx = [None for _ in range(self.num_layers)]
        self.prev_query = None
        self.curr_query = None

    def prefill_encode_partial(self, query, key, layer_idx):
        assert layer_idx >= self.num_skip_layers, f"partial topk is not enabled in layer{layer_idx}!"
        seq_len = key.shape[1]
        query = torch.abs(query)
        if self.gqa_size > 1:
            query = query.view(self.curr_batch_size, seq_len,
                               self.num_key_value_heads, self.gqa_size,
                               self.head_dim).sum(3)
        query = query.sum(1, keepdim=True)
        partial_idx = torch.topk(query,
                                 k=self.partial_dim,
                                 dim=-1,
                                 largest=True).indices
        self.partial_idx[layer_idx] = partial_idx

        key = torch.gather(key,
                           dim=-1,
                           index=partial_idx.expand(-1, seq_len, -1, -1))
        self.layer_partial_caches[layer_idx][:, :seq_len, :, :] = key
        self.layer_partial_seq_lens[layer_idx] = seq_len

    def decode_encode_partial_key(self, key, layer_idx):
        assert layer_idx >= self.num_skip_layers, f"partial topk is not enabled in layer{layer_idx}!"
        partial_idx = self.partial_idx[layer_idx]

        key = torch.gather(key, dim=-1, index=partial_idx)
        self.layer_partial_caches[
            layer_idx][:, self.layer_partial_seq_lens[layer_idx]:self.
                       layer_partial_seq_lens[layer_idx] + 1, :, :] = key
        self.layer_partial_seq_lens[layer_idx] += 1

    def decode_encode_partial_query(self, query, layer_idx):
        assert layer_idx >= self.num_skip_layers, f"partial topk is not enabled in layer{layer_idx}!"
        partial_idx = self.partial_idx[layer_idx]

        if self.gqa_size > 1:
            query = query.view(self.curr_batch_size, 1,
                               self.num_key_value_heads, self.gqa_size,
                               self.head_dim)
            query = torch.gather(query,
                                 dim=-1,
                                 index=partial_idx.unsqueeuze(-1).expand(
                                     -1, -1, -1, self.gqa_size, -1))
            query = query.view(self.curr_batch_size, 1, self.num_heads,
                               self.partial_dim)
        else:
            query = torch.gather(query, dim=-1, index=partial_idx)

        return query

    def compute_topk(self, query: torch.Tensor, layer_idx: int):
        assert layer_idx >= self.num_skip_layers, f"partial topk is not enabled in layer{layer_idx}!"
        torch.cuda.nvtx.range_push("partial score")
        score = qk_score(query, self.layer_partial_caches[layer_idx],
                         self.layer_partial_seq_lens[layer_idx])

        if self.gqa_size > 1:
            score = score.view(self.curr_batch_size, self.num_key_value_heads,
                               self.gqa_size, -1).sum(-2)
            # Simulate prefetch scenario, where the last token must be selected
            score[:, :, -1:] = torch.finfo(score.dtype).max
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("compute topk")
        # Simulate prefetch scenario, where the last token must be selected
        if self.sparse_ratio < 1:
            fetch_num = int((self.layer_partial_seq_lens[layer_idx] - 1) *
                            self.sparse_ratio) + 1
        else:
            fetch_num = min(
                int(self.sparse_ratio) + 1,
                self.layer_partial_seq_lens[layer_idx])

        # topk_indices = KVLib.batch_topk(score, fetch_num, True)
        topk_indices = torch.topk(score, fetch_num, dim=-1,
                                  largest=True).indices.int()

        torch.cuda.nvtx.range_pop()

        return topk_indices

    def get_num_skip_layers(self):
        return self.num_skip_layers

    def register_query(self, query):
        self.prev_query = self.curr_query
        self.curr_query = query

    def get_query(self):
        return self.prev_query

    def skew(self, states, layer_idx):
        states = torch.matmul(states.transpose(1, 2),
                              self.skewing_matrix[layer_idx])
        return states.transpose(1, 2).contiguous()


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
        self._cache = InfiniGenStaticCache(
            config=self.config.get_text_config(),
            num_channels=generation_config.num_channels,
            max_gpu_cache_memory_size=generation_config.max_gpu_cache_memory,
            device=device,
            dtype=self.dtype,
            layer_device_map=layer_device_map,
            sparse_ratio=generation_config.sparse_ratio,
            skewing_matrix_path=generation_config.skewing_matrix_path,
        )
        self._cache.build_cache()

    self._cache.reset(batch_size)
    cache_name = "past_key_values"
    model_kwargs[cache_name] = self._cache
