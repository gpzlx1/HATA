from typing import Dict, Optional, Union, Any
import torch
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.configuration_utils import GenerationConfig
from .kernels.triton_hash_encode import prefill_hash_encode, decode_hash_encode
from .kernels.triton_score_process import hash_score_process
import KVLib
import os


class HashStaticCache(Cache):

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
    ) -> None:
        super().__init__()

        self.max_gpu_cache_memory_size = max_gpu_cache_memory_size
        self.dtype = config.torch_dtype
        self.num_layers = config.num_hidden_layers
        self.head_dim = (config.head_dim if hasattr(config, "head_dim") else
                         config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads = (config.num_attention_heads if getattr(
            config, "num_key_value_heads", None) is None else
                                    config.num_key_value_heads)
        self.num_heads = config.num_attention_heads
        self.layer_devices = []
        self.layer_caches = []
        self.max_layer_caches = []
        self.layer_hash_caches = []
        self.max_layer_hash_caches = []
        self.layer_norm_caches = []
        self.max_layer_norm_caches = []
        self.hash_weights = []
        self.hash_rbits = hash_rbits
        assert self.hash_rbits % 32 == 0
        self.hash_dim = hash_rbits // 32

        self.num_skip_layers = num_skip_layers
        self.sparse_ratio = sparse_ratio

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

        for l in range(config.num_hidden_layers):
            if layer_device_map is not None:
                layer_device = layer_device_map[l]
            else:
                layer_device = device
            self.layer_devices.append(layer_device)

            self.layer_caches.append(None)
            self.layer_hash_caches.append(None)
            self.layer_norm_caches.append(None)

            self.max_layer_caches.append(
                torch.zeros((kv_numel, ),
                            dtype=self.dtype,
                            device=layer_device))

            if l >= self.num_skip_layers:
                self.max_layer_hash_caches.append(
                    torch.zeros((hash_numel, ),
                                dtype=torch.int32,
                                device=layer_device))
                self.max_layer_norm_caches.append(
                    torch.zeros((norm_numel, ),
                                dtype=self.dtype,
                                device=layer_device))
                if hash_weights_path is None:
                    hash_weight = torch.randn((self.head_dim, self.hash_rbits),
                                              dtype=self.dtype,
                                              device=layer_device)
                else:
                    hash_weight = torch.load(
                        os.path.join(
                            hash_weights_path,
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

        self.hash_packbit_aux_tensor = torch.pow(
            2,
            torch.arange(0,
                         32,
                         1,
                         dtype=torch.int32,
                         device=self.layer_devices[0]))

        self.query_code_buffer = None

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.seq_len

    def get_max_length(self) -> Optional[int]:
        return self.max_seq_len

    def reorder_cache(self, beam_idx: torch.LongTensor):
        raise NotImplementedError

    def key_append(self,
                   key_states: torch.Tensor,
                   layer_idx: int,
                   inc_seq_len=True):

        kv_numel = key_states.shape[0]
        q_len = kv_numel // self.curr_batch_size

        assert q_len < self.max_seq_len, "q_len should be less than max_seq_len"

        key_states = key_states.view(self.curr_batch_size, q_len,
                                     self.num_key_value_heads, self.head_dim)
        self.layer_caches[layer_idx][0, :, self.seq_len:self.seq_len +
                                     q_len, :, :] = key_states

        key_states = self.layer_caches[layer_idx][0, :, :self.seq_len +
                                                  q_len, :, :]

        if inc_seq_len and layer_idx == len(self.layer_caches) - 1:
            self.seq_len += self.get_cur_q_len()

        return key_states

    def value_proj_and_append(self,
                              hidden_states,
                              value_weightsT,
                              layer_idx,
                              inc_seq_len=False):
        hidden_size = value_weightsT.shape[0]
        hidden_states = hidden_states.view(self.curr_batch_size, -1,
                                           hidden_size)
        q_len = hidden_states.shape[1]
        self.layer_caches[layer_idx] = self.layer_caches[layer_idx].view(
            2, self.curr_batch_size, -1, hidden_size)
        torch.matmul(
            hidden_states,
            value_weightsT,
            out=self.layer_caches[layer_idx][1, :, self.seq_len:self.seq_len +
                                             q_len, :])
        self.layer_caches[layer_idx] = self.layer_caches[layer_idx].view(
            2, self.curr_batch_size, -1, self.num_key_value_heads,
            self.head_dim)

        value_states = self.layer_caches[layer_idx][1, :, :self.seq_len +
                                                    q_len, :, :]

        if inc_seq_len and layer_idx == self.num_layers - 1:
            self.seq_len += q_len

        return value_states

    def reset(self, batch_size):
        self.curr_batch_size = batch_size
        self.seq_len = 0
        self.layer_hash_seq_lens = [0 for _ in range(self.num_layers)]

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

        numel = 2 * batch_size * self.max_seq_len * self.num_key_value_heads * self.head_dim
        hash_numel = batch_size * self.max_seq_len * self.num_key_value_heads * self.hash_dim
        norm_numel = batch_size * self.max_seq_len * self.num_key_value_heads

        for i in range(self.num_layers):
            self.layer_caches[i] = self.max_layer_caches[i][:numel]
            self.layer_caches[i] = self.layer_caches[i].view(
                2, batch_size, self.max_seq_len, self.num_key_value_heads,
                self.head_dim)

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

        self.query_code_buffer = torch.empty(batch_size,
                                             1,
                                             self.num_heads,
                                             self.hash_dim,
                                             dtype=torch.int32,
                                             device=self.layer_devices[0])

    def alloc(self, q_len):
        self._rope_indptr = torch.tensor(
            [i * q_len for i in range(self.curr_batch_size + 1)],
            dtype=torch.int32,
            device=self.layer_caches[0].device)
        self._rope_offsets = torch.full((self.curr_batch_size, ),
                                        self.seq_len,
                                        dtype=torch.int32,
                                        device=self.layer_caches[0].device)
        self.cur_q_len = q_len

    def get_rope_metadata(self):
        return self._rope_indptr, self._rope_offsets

    def get_cur_q_len(self):
        return self.cur_q_len

    def get_hash_weights(self, layer_idx):
        return self.hash_weights[layer_idx]

    def prefill_encode_hash(self, key, layer_idx):
        assert layer_idx >= self.num_skip_layers, f"hash topk is not enabled in layer{layer_idx}!"
        key = key.view(self.curr_batch_size, -1, self.num_key_value_heads,
                       self.head_dim)
        seq_len = key.shape[1]
        prefill_hash_encode(
            key, self.hash_weights[layer_idx],
            self.layer_hash_caches[layer_idx][:, :seq_len, :, :],
            self.layer_norm_caches[layer_idx][:, :seq_len, :],
            self.hash_packbit_aux_tensor)
        self.layer_hash_seq_lens[layer_idx] += seq_len

    def decode_encode_hash(self, query, key, layer_idx):
        assert layer_idx >= self.num_skip_layers, f"hash topk is not enabled in layer{layer_idx}!"
        query = query.view(self.curr_batch_size, 1, self.num_key_value_heads,
                           self.head_dim)
        key = key.view(self.curr_batch_size, 1, self.num_key_value_heads,
                       self.head_dim)
        decode_hash_encode(
            key,
            self.hash_weights[layer_idx],
            self.layer_hash_caches[layer_idx]
            [:, self.layer_hash_seq_lens[layer_idx]:self.
             layer_hash_seq_lens[layer_idx] + 1, :, :],
            self.layer_norm_caches[layer_idx]
            [:, self.layer_hash_seq_lens[layer_idx]:self.
             layer_hash_seq_lens[layer_idx] + 1, :],
            query,
            self.query_code_buffer,
            self.hash_packbit_aux_tensor,
        )
        self.layer_hash_seq_lens[layer_idx] += 1

        return self.query_code_buffer

    def compute_topk(self, encoded_query: torch.Tensor, layer_idx: int):
        assert layer_idx >= self.num_skip_layers, f"hash topk is not enabled in layer{layer_idx}!"
        torch.cuda.nvtx.range_push("hash score")
        encoded_query = encoded_query.view(self.curr_batch_size,
                                           self.cur_q_len, self.num_heads,
                                           self.hash_dim)
        past_key_hash_cache = self.layer_hash_caches[
            layer_idx][:, :self.layer_hash_seq_lens[layer_idx], :, :]
        hamming_dist = KVLib.hamming_distance(past_key_hash_cache,
                                              encoded_query)
        score = hash_score_process(
            hamming_dist,
            self.layer_norm_caches[layer_idx]
            [:, :self.layer_hash_seq_lens[layer_idx], :],
            self.hash_rbits,
        )
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("compute topk")
        if self.sparse_ratio < 1:
            fetch_num = int(self.layer_hash_seq_lens[layer_idx] *
                            self.sparse_ratio)
        else:
            fetch_num = min(int(self.sparse_ratio),
                            self.layer_hash_seq_lens[layer_idx])
        topk_indices = KVLib.batch_topk(score, fetch_num, True)
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
        self._cache = HashStaticCache(
            config=self.config.get_text_config(),
            hash_rbits=generation_config.hash_rbits,
            max_gpu_cache_memory_size=generation_config.max_gpu_cache_memory,
            device=device,
            dtype=self.dtype,
            layer_device_map=layer_device_map,
            sparse_ratio=generation_config.sparse_ratio,
            hash_weights_path=generation_config.hash_weights_path,
        )

    self._cache.reset(batch_size)
    cache_name = "past_key_values"
    model_kwargs[cache_name] = self._cache
