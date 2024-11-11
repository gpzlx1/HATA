import flashinfer
import torch
import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, LlamaRotaryEmbedding
from transformers import AutoConfig

head = 32
head_dim = 128
seq_len = 1000
bs = 2

config = AutoConfig.from_pretrained(
    "/nfs/shared_LLM_model/lmsys/longchat-7b-v1.5-32k")
print(config)

rope = LlamaRotaryEmbedding(
    device='cuda',
    config=config,
)

hidden_states = torch.randn((bs, seq_len, head * head_dim),
                            device=torch.device("cuda"),
                            dtype=torch.float16)
pos = torch.arange(seq_len, device=torch.device("cuda")).unsqueeze(0)
cos, sin = rope(hidden_states, pos)
query_states = torch.randn((bs, seq_len, head, head_dim),
                           device=torch.device("cuda"),
                           dtype=torch.float16)
key_states = torch.randn((bs, seq_len, head, head_dim),
                         device=torch.device("cuda"),
                         dtype=torch.float16)
hf_q, hf_k = apply_rotary_pos_emb(query_states,
                                  key_states,
                                  cos,
                                  sin,
                                  unsqueeze_dim=2)
hf_q = hf_q.view(-1)
hf_k = hf_k.view(-1)

# flashinfer

indptr = torch.tensor([i * seq_len for i in range(bs + 1)],
                      dtype=torch.int32,
                      device="cuda:0")
offsets = torch.full((bs, ), 0, dtype=torch.int32, device="cuda:0")
fl_q, fl_k = flashinfer.apply_rope(query_states.view(-1, head, head_dim),
                                   key_states.view(-1, head, head_dim),
                                   indptr,
                                   offsets,
                                   interleave=False,
                                   rope_scale=8.0,
                                   rope_theta=10000)
fl_q = fl_q.view(-1)
fl_k = fl_k.view(-1)

# print(hf_q.view(bs, seq_len, head, head_dim))
# print(fl_q.view(bs, seq_len, head, head_dim))

print((fl_q - hf_q).max())
print((fl_q - hf_q).min())
