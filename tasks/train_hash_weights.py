import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
from transformers import AutoModelForCausalLM
import argparse


class BinarySTE(nn.Module):

    def __init__(self):
        super(BinarySTE, self).__init__()

    def forward(self, x):
        return (x > 0).float()

    def backward(self, grad_output):
        return grad_output


class BinaryStochastic(nn.Module):

    def __init__(self):
        super(BinaryStochastic, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.training:
            return (x.tanh() + 1) / 2
        else:
            return (x >= 0).float()


class MyModel(torch.nn.Module):

    def __init__(self, hash_weight, q_weight, k_weight, rbit, num_heads,
                 num_kv_heads, head_dim, batch_size):
        super(MyModel, self).__init__()
        self.hash_weight = hash_weight
        torch.nn.init.xavier_uniform_(self.hash_weight)
        self.q_weight = q_weight
        self.k_weight = k_weight
        self.rbit = rbit
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.gqa_size = self.num_heads // self.num_kv_heads
        self.head_dim = head_dim
        self.batch_size = batch_size

        self.bn = BinaryStochastic()

    def forward(self, q_hidden, k_hidden):
        q_states = (q_hidden @ self.q_weight).view(self.batch_size,
                                                   self.num_heads,
                                                   self.head_dim)
        k_states = (k_hidden @ self.k_weight).view(self.batch_size,
                                                   self.num_kv_heads, 1,
                                                   self.head_dim)
        k_states = k_states.expand(-1, -1, self.gqa_size,
                                   -1).reshape(-1, self.num_heads,
                                               self.head_dim)
        attn_weights = ((q_states[:, :, None, :] @ k_states[:, :, :, None])
                        ).view(-1) / math.sqrt(self.head_dim)

        q_code = ((q_states @ self.hash_weight))
        k_code = ((k_states @ self.hash_weight))

        q_code = self.bn(q_code)
        k_code = self.bn(k_code)

        q_norm = q_states.norm(dim=-1)
        k_norm = k_states.norm(dim=-1)

        ham_dist = ((q_code - k_code).abs().sum(dim=-1))

        ham_weights = ((1 - 2 * ham_dist / rbit) * q_norm *
                       k_norm).view(-1) / math.sqrt(self.head_dim)

        return attn_weights, ham_weights


def train_hash_weights(
        q_weight,
        k_weight,
        rbit,
        num_heads,
        num_kv_heads,
        head_dim,
        #    seq_len=1000,
        batch_size=4000,
        iters=4000,
        schedule_iters=100):
    dtype, device = q_weight.dtype, q_weight.device
    q_weight = q_weight.to(dtype)
    k_weight = k_weight.to(dtype)
    hash_weight = torch.normal(0,
                               std,
                               size=(head_dim, rbit),
                               device=device,
                               dtype=dtype,
                               requires_grad=True)
    model = MyModel(hash_weight, q_weight, k_weight, rbit, num_heads,
                    num_kv_heads, head_dim, batch_size)

    loss_func = nn.MSELoss()
    # loss_func = nn.MSELoss()
    optimizer = optim.SGD([hash_weight], lr=12)
    # loss_func = nn.CrossEntropyLoss()
    # optimizer = optim.Adam([hash_weight], lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    eval_loss_func = nn.MSELoss()

    for it in range(iters):
        model.train()
        q_hidden = torch.randn((batch_size, num_heads * head_dim),
                               dtype=dtype,
                               device=device)
        k_hidden = torch.randn((batch_size, num_heads * head_dim),
                               dtype=dtype,
                               device=device)

        attn_weights, ham_weights = model(q_hidden, k_hidden)

        loss = loss_func(attn_weights, ham_weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (it + 1) % schedule_iters == 0:
            with torch.no_grad():
                model.eval()
                attn_weights, ham_weights = model(q_hidden, k_hidden)
                eval_loss = eval_loss_func(attn_weights, ham_weights)
                print(f"Iter: {it:5d} loss: {loss}, eval_loss: {eval_loss}")
            scheduler.step()

    return hash_weight


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="/nfs/shared_LLM_model/lmsys/longchat-7b-v1.5-32k")
    parser.add_argument("--rbit", type=int, default=256)
    parser.add_argument("--std", type=float, default=1.0)
    parser.add_argument("--save_path", type=str, default=".")

    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 torch_dtype=torch.float16)
    model_name = os.path.basename(os.path.normpath(args.model_path))
    rbit = args.rbit
    std = args.std

    q_weights = []
    k_weights = []
    hash_weights = []

    head_dim = model.model.layers[0].self_attn.head_dim
    num_heads = model.model.layers[0].self_attn.num_heads
    num_kv_heads = model.model.layers[0].self_attn.num_key_value_heads
    num_layers = len(model.model.layers)

    for l in range(num_layers):
        q_weights.append(
            model.model.layers[l].self_attn.q_proj.weight.detach().transpose(
                0, 1).to("cuda"))
        k_weights.append(
            model.model.layers[l].self_attn.k_proj.weight.detach().transpose(
                0, 1).to("cuda"))

    del model

    for l in range(32):
        if l <= 1:
            continue
        print(f"Layer {l:2d}")
        hash_weight = train_hash_weights(q_weights[l], k_weights[l], rbit,
                                         num_heads, num_kv_heads, head_dim)

        save_path = os.path.join(args.save_path,
                                 f"hash_weight_layer_{l:02d}.pt")
        torch.save(hash_weight.cpu(), save_path)
