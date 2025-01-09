import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
from transformers import AutoModelForCausalLM
import argparse


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

    def __init__(self, hash_weight, proj_weight, rbit, num_heads, num_kv_heads,
                 head_dim, batch_size, with_norm):
        super(MyModel, self).__init__()
        self.hash_weight = hash_weight
        torch.nn.init.xavier_uniform_(self.hash_weight)
        self.proj_weight = proj_weight
        self.rbit = rbit
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.gqa_size = self.num_heads // self.num_kv_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.with_norm = with_norm

        self.bn = BinaryStochastic()

    def forward(self, hidden):
        qkv = self.proj_weight(hidden)
        (q_states, k_states, _) = qkv.split(
            [
                self.num_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        q_states = q_states.view(self.batch_size, self.num_heads,
                                 self.head_dim)
        k_states = k_states.view(self.batch_size, self.num_kv_heads, 1,
                                 self.head_dim)
        k_states = k_states.expand(-1, -1, self.gqa_size,
                                   -1).reshape(-1, self.num_heads,
                                               self.head_dim)

        if self.with_norm:
            attn_weights = ((q_states[:, :, None, :] @ k_states[:, :, :, None])
                            ).view(-1) / math.sqrt(self.head_dim)
        else:
            attn_weights = torch.cosine_similarity(q_states, k_states,
                                                   dim=-1).view(-1)

        q_code = ((q_states @ self.hash_weight))
        k_code = ((k_states @ self.hash_weight))

        q_code = self.bn(q_code)
        k_code = self.bn(k_code)

        ham_dist = ((q_code - k_code).abs().sum(dim=-1))

        ham_weights = (1 - 2 * ham_dist / rbit).to(q_code.dtype)

        if self.with_norm:
            q_norm = q_states.norm(dim=-1)
            k_norm = k_states.norm(dim=-1)
            ham_weights = (ham_weights * q_norm * k_norm) / math.sqrt(
                self.head_dim)

        ham_weights = ham_weights.view(-1)

        # drop_mask = ~torch.isnan(attn_weights) & ~torch.isinf(
        #     attn_weights) & ~torch.isneginf(attn_weights)
        # attn_weights = attn_weights[drop_mask]
        # ham_weights = ham_weights[drop_mask]

        return attn_weights, ham_weights


def train_hash_weights(
    proj_weight,
    rbit,
    num_heads,
    num_kv_heads,
    head_dim,
    lr=12,
    dtype=torch.float16,
    device="cuda",
    with_norm=True,
    batch_size=4000,
    iters=4000,
    schedule_iters=100,
):
    proj_weight = proj_weight.to(dtype)
    hash_weight = torch.normal(0,
                               std,
                               size=(head_dim, rbit),
                               device=device,
                               dtype=dtype,
                               requires_grad=True)
    model = MyModel(hash_weight, proj_weight, rbit, num_heads, num_kv_heads,
                    head_dim, batch_size, with_norm)

    loss_func = nn.MSELoss()
    # loss_func = nn.MSELoss()
    optimizer = optim.SGD([hash_weight], lr=lr)
    # loss_func = nn.CrossEntropyLoss()
    # optimizer = optim.Adam([hash_weight], lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    eval_loss_func = nn.MSELoss()

    for it in range(iters):
        model.train()
        hidden = torch.randn((batch_size, num_heads * head_dim),
                             dtype=dtype,
                             device=device)

        attn_weights, ham_weights = model(hidden)

        loss = loss_func(attn_weights, ham_weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (it + 1) % schedule_iters == 0:
            with torch.no_grad():
                model.eval()
                attn_weights, ham_weights = model(hidden)
                eval_loss = eval_loss_func(attn_weights, ham_weights)
                print(f"Iter: {it:5d} loss: {loss}, eval_loss: {eval_loss}")
            scheduler.step()

    return hash_weight


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        type=str,
                        default="/nfs/shared_LLM_model/THUDM/glm-4-9b-chat")
    parser.add_argument("--rbit", type=int, default=256)
    parser.add_argument("--std", type=float, default=1.0)
    parser.add_argument("--save_path", type=str, default=".")
    parser.add_argument("--with_norm", action="store_true", default=False)

    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 torch_dtype=torch.float16,
                                                 trust_remote_code=True)
    rbit = args.rbit
    std = args.std

    proj_weights = []
    hash_weights = []

    head_dim = model.transformer.encoder.layers[
        0].self_attention.hidden_size_per_attention_head
    num_heads = model.transformer.encoder.layers[
        0].self_attention.num_attention_heads_per_partition
    num_kv_heads = model.transformer.encoder.layers[
        0].self_attention.num_multi_query_groups_per_partition
    num_layers = model.transformer.encoder.num_layers

    for l in range(num_layers):
        proj_weights.append(model.transformer.encoder.layers[0].self_attention.
                            query_key_value.cuda())

    del model

    for l in range(num_layers):
        if l <= 1:
            continue
        print(f"Layer {l:2d}")
        hash_weight = train_hash_weights(proj_weights[l],
                                         rbit,
                                         num_heads,
                                         num_kv_heads,
                                         head_dim,
                                         lr=12,
                                         with_norm=args.with_norm)

        save_path = os.path.join(args.save_path,
                                 f"hash_weight_layer_{l:02d}.pt")
        torch.save(hash_weight.cpu(), save_path)
