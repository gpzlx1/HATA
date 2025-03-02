from typing import Dict, Optional, Union, Any
import torch
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.configuration_utils import GenerationConfig


class CustomStaticCache(Cache):

    def __init__(
        self,
        config: PretrainedConfig,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
        max_gpu_cache_memory_size: int = 1000000000,  # 0.93 GB
        layer_device_map: Optional[Dict[int, Union[str, torch.device,
                                                   int]]] = None,
    ) -> None:
        super().__init__()

        self.max_gpu_cache_memory_size = max_gpu_cache_memory_size
        self.dtype = config.torch_dtype
        self.num_layers = config.num_hidden_layers

        self.nope_dim = config.qk_nope_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.lora_dim = config.kv_lora_rank

        self.layer_devices = []
        for l in range(self.num_layers):
            if layer_device_map is not None:
                layer_device = layer_device_map[l]
                self.layer_devices.append(layer_device)
            else:
                layer_device = torch.device(device)
                self.layer_devices.append(layer_device.index)
        self.unique_devices = set(self.layer_devices)
        self.rope_metadata = {}

    def build_cache(self):
        self.layer_kpe_caches = []
        self.layer_lora_caches = []
        self.max_layer_kpe_caches = []
        self.max_layer_lora_caches = []

        self.each_layer_max_gpu_cache = self.max_gpu_cache_memory_size / self.num_layers
        numel = int(self.each_layer_max_gpu_cache / self.dtype.itemsize)
        self.each_layer_kpe_numel = int(numel * self.rope_dim /
                                        (self.rope_dim + self.lora_dim))
        self.each_layer_lora_numel = int(numel * self.lora_dim /
                                         (self.rope_dim + self.lora_dim))

        for l in range(self.num_layers):
            layer_device = self.layer_devices[l]
            self.layer_kpe_caches.append(None)
            self.max_layer_kpe_caches.append(
                torch.zeros((self.each_layer_kpe_numel, ),
                            dtype=self.dtype,
                            device=layer_device))
            self.layer_lora_caches.append(None)
            self.max_layer_lora_caches.append(
                torch.zeros((self.each_layer_lora_numel, ),
                            dtype=self.dtype,
                            device=layer_device))

        self.seq_len = 0
        self.max_seq_len = 0
        self.curr_batch_size = 0

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.seq_len

    def get_max_length(self) -> Optional[int]:
        return self.max_seq_len

    def reorder_cache(self, beam_idx: torch.LongTensor):
        raise NotImplementedError

    def append(self,
               key_states: torch.Tensor,
               layer_idx: int,
               type="key",
               inc_seq_len=True):

        q_len = self.cur_q_len
        assert q_len < self.max_seq_len, "q_len should be less than max_seq_len"

        if type == "key":
            key_states = key_states.view(self.curr_batch_size, q_len,
                                         self.rope_dim)
            self.layer_kpe_caches[layer_idx][:, self.seq_len:self.seq_len +
                                             q_len, :] = key_states
            key_states = self.layer_kpe_caches[layer_idx][:, :self.seq_len +
                                                          q_len, :]
        else:
            key_states = key_states.view(self.curr_batch_size, q_len,
                                         self.lora_dim)
            self.layer_lora_caches[layer_idx][:, self.seq_len:self.seq_len +
                                              q_len, :] = key_states
            key_states = self.layer_lora_caches[layer_idx][:, :self.seq_len +
                                                           q_len, :]

        if inc_seq_len and layer_idx == self.num_layers - 1:
            self.seq_len += self.get_cur_q_len()

        return key_states

    def reset(self, batch_size):
        self.curr_batch_size = batch_size
        self.seq_len = 0

        max_kpe_seq_len = self.each_layer_kpe_numel // (self.rope_dim *
                                                        self.curr_batch_size)
        max_lora_seq_len = self.each_layer_lora_numel // (self.lora_dim *
                                                          self.curr_batch_size)
        self.max_seq_len = min(max_kpe_seq_len, max_lora_seq_len)
        kpe_numel = batch_size * self.max_seq_len * self.rope_dim
        lora_numel = batch_size * self.max_seq_len * self.lora_dim

        for i in range(self.num_layers):
            self.layer_kpe_caches[i] = self.max_layer_kpe_caches[
                i][:kpe_numel].view(batch_size, self.max_seq_len,
                                    self.rope_dim)
            self.layer_lora_caches[i] = self.max_layer_lora_caches[
                i][:lora_numel].view(batch_size, self.max_seq_len,
                                     self.lora_dim)

    def alloc(self, q_len):
        for device in self.unique_devices:
            rope_indptr = torch.arange(
                (self.curr_batch_size + 1), dtype=torch.int32,
                device=device) * q_len
            rope_offsets = torch.full((self.curr_batch_size, ),
                                      self.seq_len,
                                      dtype=torch.int32,
                                      device=device)
            self.rope_metadata[device] = (rope_indptr, rope_offsets)
        self.cur_q_len = q_len

    def get_rope_metadata(self, device=None):
        if device is None:
            return self.rope_metadata[self.layer_devices[0]]
        else:
            if hasattr(device, "index"):
                return self.rope_metadata[device.index]
            else:
                return self.rope_metadata[device]

    def get_cur_q_len(self):
        return self.cur_q_len


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
        self._cache = CustomStaticCache(
            config=self.config.get_text_config(),
            max_gpu_cache_memory_size=generation_config.max_gpu_cache_memory,
            device=device,
            dtype=self.dtype,
            layer_device_map=layer_device_map,
        )
        self._cache.build_cache()

    self._cache.reset(batch_size)
    cache_name = "past_key_values"
    model_kwargs[cache_name] = self._cache
