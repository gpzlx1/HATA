from typing import Dict, Optional, Union, Any
import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.configuration_utils import GenerationConfig
from .kvcache_fa import CustomStaticCache
from .kernels.triton_qk_score import loki_qk_score
import KVLib
import os


def qk_score(query, key, seq_len, partial_dim):
    b, _, hk, _ = key.shape
    h = query.shape[-2]
    gqa = h // hk

    query = query[:, :, :, :partial_dim].transpose(1, 2).transpose(-1, -2)
    key = key[:, :seq_len, :, :partial_dim].transpose(
        1, 2).unsqueeze(2).expand(-1, -1, gqa, -1,
                                  -1).reshape(b, h, seq_len, partial_dim)
    score = key @ query
    score = score.view(b, hk, gqa, -1).sum(2)
    return score


class LokiStaticCache(CustomStaticCache):

    def __init__(
        self,
        config: PretrainedConfig,
        num_channels: int = 32,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
        max_gpu_cache_memory_size: int = 1000000000,  # 0.93 GB
        layer_device_map: Optional[Dict[int, Union[str, torch.device,
                                                   int]]] = None,
        sparse_ratio: float = 0.1,
        num_skip_layers: int = 2,
        aux_data_path: str = None,
        num_sink: int = 0,
        num_recent: int = 0,
    ) -> None:
        super().__init__(config, device, dtype, max_gpu_cache_memory_size,
                         layer_device_map)
        self.partial_dim = num_channels
        self.sparse_ratio = sparse_ratio
        self.aux_data_path = aux_data_path
        self.num_skip_layers = num_skip_layers
        self.gqa_size = self.num_heads // self.num_key_value_heads
        self.num_sink = num_sink
        self.num_recent = num_recent

    def load_aux_data(self):
        self.pca_matrix = []
        self.pca_expand_matrix = []
        for l in range(self.num_layers):
            if l < self.num_skip_layers:
                self.pca_matrix.append(None)
                self.pca_expand_matrix.append(None)
            else:
                pca = torch.load(
                    os.path.join(
                        self.aux_data_path,
                        f"pca_components/pca_components_layer_{l:02d}.pt"))
                pca = pca.view(1, self.num_key_value_heads, self.head_dim,
                               self.head_dim).transpose(2, 3).contiguous().to(
                                   self.dtype).to(self.layer_devices[l])
                self.pca_matrix.append(pca)
                if self.gqa_size > 1:
                    pca_expand = pca.unsqueeze(2).expand(
                        -1, -1, self.gqa_size, -1,
                        -1).reshape(1, self.num_heads, self.head_dim,
                                    self.head_dim)
                    self.pca_expand_matrix.append(pca_expand)
                else:
                    self.pca_expand_matrix.append(None)
        if self.gqa_size > 1:
            self.aux_data_size = (
                self.num_key_value_heads + self.num_heads
            ) * self.head_dim * self.head_dim * self.dtype.itemsize * (
                self.num_layers - self.num_skip_layers)
        else:
            self.aux_data_size = self.num_key_value_heads * self.head_dim * self.head_dim * self.dtype.itemsize * (
                self.num_layers - self.num_skip_layers)

    def build_cache(self):
        self.load_aux_data()
        self.max_gpu_cache_memory_size -= self.aux_data_size

        self.layer_caches = []
        self.max_layer_caches = []

        each_layer_max_gpu_cache = self.max_gpu_cache_memory_size / self.num_layers
        numel = int(each_layer_max_gpu_cache / self.dtype.itemsize)
        self.each_layer_max_numel = numel

        for l in range(self.num_layers):
            layer_device = self.layer_devices[l]
            self.layer_caches.append(None)
            self.max_layer_caches.append(
                torch.zeros((numel, ), dtype=self.dtype, device=layer_device))

        self.max_seq_len = 0
        self.curr_batch_size = 0
        self.seq_len = 0

    def encode_key(self, key, layer_idx):
        assert layer_idx >= self.num_skip_layers, f"pca is not enabled in layer{layer_idx}!"
        seq_len = key.shape[1]
        if seq_len > 1:
            key = key.transpose(
                1, 2)  # (b, h, s, d) to reduce cuda mem used during matmul
            key = torch.matmul(key, self.pca_matrix[layer_idx])
            key = key.transpose(1, 2)  # (b, s, h, d)
        else:
            key = key.view(self.curr_batch_size, self.num_key_value_heads, 1,
                           self.head_dim)
            key = torch.matmul(key, self.pca_matrix[layer_idx])
            key = key.view(self.curr_batch_size, 1, self.num_key_value_heads,
                           self.head_dim)
        return key

    def encode_query(self, query, layer_idx):
        assert layer_idx >= self.num_skip_layers, f"pca is not enabled in layer{layer_idx}!"
        query = query.view(self.curr_batch_size, self.num_heads, 1,
                           self.head_dim)
        if self.gqa_size > 1:
            query = torch.matmul(query, self.pca_expand_matrix[layer_idx])
        else:
            query = torch.matmul(query, self.pca_matrix[layer_idx])
        query = query.view(self.curr_batch_size, 1, self.num_heads,
                           self.head_dim)
        return query

    def compute_topk(self, query: torch.Tensor, layer_idx: int):
        assert layer_idx >= self.num_skip_layers, f"partial topk is not enabled in layer{layer_idx}!"

        kvcache_len = self.seq_len + 1 if layer_idx != self.num_layers - 1 else self.seq_len

        torch.cuda.nvtx.range_push("partial score")
        score = loki_qk_score(query, self.layer_caches[layer_idx][0],
                              kvcache_len, self.partial_dim)
        # score shape = [batch_size, head_num, seq_len]
        score = torch.softmax(score.to(torch.float32),
                              dim=-1).to(torch.float16)
        score = score.view(self.curr_batch_size, self.num_key_value_heads,
                           self.gqa_size, kvcache_len)
        score = torch.sum(score, dim=2)

        if self.num_sink > 0:
            score[:, :, :self.num_sink] = torch.finfo(score.dtype).max
        if self.num_recent > 0:
            score[:, :, -self.num_recent:] = torch.finfo(score.dtype).max
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("compute topk")
        if self.sparse_ratio < 1:
            fetch_num = int(kvcache_len * self.sparse_ratio)
        else:
            fetch_num = min(int(self.sparse_ratio), kvcache_len)
        topk_indices = KVLib.batch_topk(score, fetch_num, True)
        # topk_indices = torch.topk(score, fetch_num, dim=-1,
        #                           largest=True).indices.int()
        torch.cuda.nvtx.range_pop()

        return topk_indices

    def get_num_skip_layers(self):
        return self.num_skip_layers


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
        self._cache = LokiStaticCache(
            config=self.config.get_text_config(),
            num_channels=generation_config.num_channels,
            max_gpu_cache_memory_size=generation_config.max_gpu_cache_memory,
            device=device,
            dtype=self.dtype,
            layer_device_map=layer_device_map,
            sparse_ratio=generation_config.sparse_ratio,
            aux_data_path=generation_config.aux_data_path,
            num_sink=generation_config.num_sink,
            num_recent=generation_config.num_recent,
        )
        self._cache.build_cache()

    self._cache.reset(batch_size)
    cache_name = "past_key_values"
    model_kwargs[cache_name] = self._cache
