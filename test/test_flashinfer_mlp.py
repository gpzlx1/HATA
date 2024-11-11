import flashinfer
import torch

import torch.nn as nn
import transformers
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRotaryEmbedding
from transformers import AutoConfig




from flashinfer.activation import silu_and_mul

import torch.nn as nn

class SiLUAndMul(nn.Module):
    def __init__(self):
        super(SiLUAndMul, self).__init__()
    
    def forward(self, x):
        return silu_and_mul(x)

class CustomerLlamaMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Linear(self.hidden_size,
                                      self.intermediate_size * 2,
                                      bias=config.mlp_bias, dtype=torch.float16)
        self.down_proj = nn.Linear(self.intermediate_size,
                                   self.hidden_size,
                                   bias=config.mlp_bias, dtype=torch.float16)
        self.act_fn = SiLUAndMul()

    def load_weights_from_hf(self, hf_llama_mlp: LlamaMLP):
        self.gate_up_proj.weight.data[:self.
                                      intermediate_size, :] = hf_llama_mlp.gate_proj.weight.data
        self.gate_up_proj.weight.data[
            self.intermediate_size:, :] = hf_llama_mlp.up_proj.weight.data
        self.down_proj.weight.data[:, :] = hf_llama_mlp.down_proj.weight.data

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x
    

config = AutoConfig.from_pretrained(
    "/nfs/shared_LLM_model/lmsys/longchat-7b-v1.5-32k")
print(config)

hf_mlp = LlamaMLP(config).cuda().to(torch.float16)
print(hf_mlp)

data = torch.randn((100, 4096), device='cuda', dtype=torch.float16)

hf_output = hf_mlp(data)
print(hf_output)


c_mlp = CustomerLlamaMLP(config).cuda()
c_mlp.load_weights_from_hf(hf_mlp)
c_output = c_mlp(data)

print(c_output)

diff = (hf_output - c_output).abs().max()
print(diff)