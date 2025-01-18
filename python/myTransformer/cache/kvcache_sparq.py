from typing import Dict, Optional, Union, Any
import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.configuration_utils import GenerationConfig
from .kvcache_fa import CustomStaticCache
from .kernels.triton_qk_score import sparq_qk_score
import KVLib
import os


class SparQStaticCache(CustomStaticCache):

    def __init__(
        self,
        config: PretrainedConfig,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
        max_gpu_cache_memory_size: int = 1000000000,  # 0.93 GB
        layer_device_map: Optional[Dict[int, Union[str, torch.device,
                                                   int]]] = None,
        sparse_ratio: float = 0.1,
        r_channel: int = 32,
        num_skip_layers: int = 2,
        num_sink: int = 0,
        num_recent: int = 0,
    ) -> None:
        super().__init__(config, device, dtype, max_gpu_cache_memory_size,
                         layer_device_map)
        self.sparse_ratio = sparse_ratio
        self.r_channel = r_channel
        self.num_skip_layers = num_skip_layers
        self.gqa_size = self.num_heads // self.num_key_value_heads
        self.num_sink = num_sink
        self.num_recent = num_recent

    def compute_topk(self, query: torch.Tensor, layer_idx: int):
        assert layer_idx >= self.num_skip_layers, f"partial topk is not enabled in layer{layer_idx}!"
        torch.cuda.nvtx.range_push("partial score")
        query_ = query.abs()
        if self.gqa_size > 1:
            query_ = query_.view(self.curr_batch_size, -1,
                                 self.num_key_value_heads, self.gqa_size,
                                 self.head_dim).sum(3)
        channel_index = torch.topk(query_, self.r_channel, dim=-1).indices
        score = sparq_qk_score(query, self.layer_caches[layer_idx][0],
                               self.seq_len, channel_index)
        if self.num_sink > 0:
            score[:, :, :self.num_sink] = torch.finfo(score.dtype).max
        if self.num_recent > 0:
            score[:, :, -self.num_recent:] = torch.finfo(score.dtype).max
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("compute topk")
        if self.sparse_ratio < 1:
            fetch_num = int(self.seq_len * self.sparse_ratio)
        else:
            fetch_num = min(int(self.sparse_ratio), self.seq_len)
        # topk_indices = KVLib.batch_topk(score, fetch_num, True)
        topk_indices = torch.topk(score, fetch_num, dim=-1,
                                  largest=True).indices.int()
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
        self._cache = SparQStaticCache(
            config=self.config.get_text_config(),
            max_gpu_cache_memory_size=generation_config.max_gpu_cache_memory,
            device=device,
            dtype=self.dtype,
            layer_device_map=layer_device_map,
            sparse_ratio=generation_config.sparse_ratio,
            r_channel=generation_config.r_channel,
            num_sink=generation_config.num_sink,
            num_recent=generation_config.num_recent,
        )
        self._cache.build_cache()

    self._cache.reset(batch_size)
    cache_name = "past_key_values"
    model_kwargs[cache_name] = self._cache
