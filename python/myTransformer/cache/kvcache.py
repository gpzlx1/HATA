from typing import Dict, Optional, Union, Any
import torch
from flashinfer import page
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.configuration_utils import GenerationConfig

from .allocator import PageAllocator


class PagedCache(Cache):

    def __init__(
        self,
        config: PretrainedConfig,
        page_num: int,
        page_size: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        max_batch_size: Optional[int] = None,
        layer_device_map: Optional[Dict[int, Union[str, torch.device,
                                                   int]]] = None,
    ) -> None:
        super().__init__()

        self.page_size = page_size
        self.page_num = page_num
        self.dtype = dtype
        self.num_layers = config.num_hidden_layers
        self.head_dim = (config.head_dim if hasattr(config, "head_dim") else
                         config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads = (config.num_attention_heads if getattr(
            config, "num_key_value_heads", None) is None else
                                    config.num_key_value_heads)
        self.layer_devices = []
        self.layer_caches = []
        self.curr_batch_size = 0
        self.seq_len = 0

        for l in range(config.num_hidden_layers):
            if layer_device_map is not None:
                layer_device = layer_device_map[l]
            else:
                layer_device = device
            self.layer_devices.append(layer_device)
            self.layer_caches.append(
                torch.zeros((page_num, 2, page_size, self.num_key_value_heads,
                             self.head_dim),
                            dtype=self.dtype,
                            device=layer_device))

        self.allocator = PageAllocator(self.page_num, self.page_size,
                                       self.layer_devices[0])

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.seq_len

    def get_max_length(self) -> Optional[int]:
        return (self.page_size * self.page_num) // self.curr_batch_size

    def reorder_cache(self, beam_idx: torch.LongTensor):
        raise NotImplementedError(
            "PagedCache does not support beam search now.")

    def update(self,
               key_states: torch.Tensor,
               value_states: torch.Tensor,
               layer_idx: int,
               cache_kwargs: Optional[Dict[str, Any]] = None):
        """
        flashinfer's page kvcache is designed for variable-length sequences batch
        however, hf assumes fixed-length sequences batch
        currently, we only support fixed-length sequences batch
        """

        append_indptr = self.get_append_indptr()
        kv_indptr, kv_indices, kv_last_lens = self.get_attn_metadata()
        page.append_paged_kv_cache(
            key_states,
            value_states,
            append_indptr,
            self.layer_caches[layer_idx],
            kv_indices.to(self.layer_devices[layer_idx]),
            kv_indptr.to(self.layer_devices[layer_idx]),
            kv_last_lens.to(self.layer_devices[layer_idx]),
            kv_layout="NHD")

        if layer_idx == len(self.layer_caches) - 1:
            self.seq_len += self.get_cur_q_len()

        return self.layer_caches[layer_idx]

    def alloc(self, batch_size, seq_len):
        for i in range(batch_size):
            self.allocator.alloc(i, seq_len)
        id_list = [i for i in range(batch_size)]
        indptr, indices, last_lens = self.allocator.get_metadata(id_list)
        self._kv_indptr = indptr
        self._kv_indices = indices
        self._kv_last_lens = last_lens
        self._rope_indptr = torch.tensor(
            [i * seq_len for i in range(batch_size + 1)],
            dtype=torch.int32,
            device=self.allocator.device)
        self._rope_offsets = torch.full((batch_size, ),
                                        self.seq_len,
                                        dtype=torch.int32,
                                        device=self.allocator.device)
        self._cur_batch_size = batch_size
        self._cur_q_len = seq_len
        self._append_indptr = torch.arange(0, (batch_size + 1) * seq_len,
                                           seq_len,
                                           dtype=torch.int32,
                                           device=self.allocator.device)

    def get_rope_metadata(self):
        return self._rope_indptr, self._rope_offsets

    def get_attn_metadata(self):
        return self._kv_indptr, self._kv_indices, self._kv_last_lens

    def get_cur_batch_size(self):
        return self._cur_batch_size

    def get_cur_q_len(self):
        return self._cur_q_len

    def get_append_indptr(self):
        return self._append_indptr

    def reset(self, batch_size):
        # reset metadata for flashinfer's page management
        self.curr_batch_size = batch_size
        self.seq_len = 0
        self.allocator.reset()

    def get_page_size(self):
        return self.page_size

    def get_page_num(self):
        return self.page_num


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
    if hasattr(self, "_cache"):
        self._cache.reset(batch_size)
    else:

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
        self._cache = PagedCache(
            self.config.get_text_config(),
            generation_config.page_num,
            generation_config.page_size,
            device=device,
            dtype=self.dtype,
            layer_device_map=layer_device_map,
        )

    cache_name = "past_key_values"
    model_kwargs[cache_name] = self._cache
