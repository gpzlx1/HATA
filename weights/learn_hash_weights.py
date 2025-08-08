import torch
import os
import glob
import argparse
import numpy as np
import random
import torch.multiprocessing as mp


def set_args(parser: argparse.ArgumentParser):
    parser.add_argument("--dataset_path",
                        type=str,
                        default="/mnt/ramdisk/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--save_path",
        type=str,
        default=
        "/root/workspace/myoffloading/model_weights_v4/Meta-Llama-3.1-8B-Instruct-128-v01"
    )
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--num_skip_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--num_kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--mp_num", type=int, default=8)
    parser.add_argument("--rbit", type=int, default=128)
    parser.add_argument("--chunk_num", type=int,
                        default=2)  # number of chunks learned in one epoch
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--train_iters", type=int, default=25)  # in one epoch
    parser.add_argument("--rep_iters", type=int, default=100)
    parser.add_argument("--sch_iters", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--lambdda", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.1)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


class HashTrainer(torch.nn.Module):

    def __init__(self,
                 num_heads,
                 num_kv_heads,
                 head_dim,
                 rbit,
                 save_path,
                 layer_idx,
                 dtype=torch.bfloat16,
                 device="cuda",
                 lr=0.1,
                 epsilon=0.01,
                 lambdda=1.0,
                 eta=1.0,
                 sigma=0.1):
        super(HashTrainer, self).__init__()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.gqa_size = num_heads // num_kv_heads
        self.head_dim = head_dim

        self.dtype = dtype
        self.device = device

        self.rbit = rbit
        self.hash_weight = torch.randn((num_kv_heads, head_dim, self.rbit),
                                       requires_grad=True,
                                       dtype=dtype,
                                       device=device)

        self.optimizer = torch.optim.SGD([self.hash_weight],
                                         lr=lr,
                                         weight_decay=1e-6,
                                         momentum=0.9)

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                gamma=0.998)

        self.save_path = os.path.join(save_path,
                                      f"hash_weight_layer_{layer_idx:02d}.pt")

        self.epsilon = epsilon
        self.lambdda = lambdda
        self.eta = eta
        self.sigma = sigma

    def _loss_func(self, q, k, score):
        num_items = q.shape[1]

        # calculate hash
        q = q.view(self.num_kv_heads, self.gqa_size, num_items, self.head_dim)
        k = k.view(self.num_kv_heads, 1, num_items,
                   self.head_dim).expand(-1, self.gqa_size, -1, -1)
        hash_weight = self.hash_weight.unsqueeze(1)
        q_hash = q @ hash_weight
        k_hash = k @ hash_weight
        q_hash = q_hash.view(self.num_heads, num_items, self.rbit)
        k_hash = k_hash.view(self.num_heads, num_items, self.rbit)
        q_hash = 2 * torch.sigmoid(self.sigma * q_hash) - 1
        k_hash = 2 * torch.sigmoid(self.sigma * k_hash) - 1

        # similarity_loss
        hash_similarity = torch.pow(torch.norm(q_hash - k_hash, dim=-1),
                                    2)  # (#heads, num_items)
        similarity_loss = self.epsilon * torch.sum(
            score * hash_similarity) / num_items

        # balance_loss
        balance_loss = (q_hash + k_hash).sum(
            1, keepdim=True) / num_items / self.rbit  # (#heads, 1, rbit)
        balance_loss = self.lambdda * (
            balance_loss @ balance_loss.transpose(-1, -2))
        balance_loss = balance_loss.sum()

        # decorrelattion_loss
        eye = torch.eye(self.hash_weight.shape[2],
                        device=q.device,
                        requires_grad=False).unsqueeze(0)
        decorrelattion_loss = self.hash_weight.transpose(-1,
                                                         -2) @ self.hash_weight
        decorrelattion_loss = torch.norm(decorrelattion_loss - eye)
        decorrelattion_loss = self.eta * decorrelattion_loss

        # final loss
        loss = similarity_loss + balance_loss + decorrelattion_loss

        return loss, similarity_loss, balance_loss, decorrelattion_loss

    def train_for_one_step(self, q, k, score):
        with torch.enable_grad():
            (
                loss,
                similarity_loss,
                balance_loss,
                decorrelattion_loss,
            ) = self._loss_func(q, k, score)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item(), similarity_loss.item(), balance_loss.item(
        ), decorrelattion_loss.item()

    def save(self):
        torch.save(self.hash_weight.cpu(), self.save_path)


def load_chunks(path, layer_idx, num_chunks, dst_device, shuffle=True):
    q_chunks = glob.glob(
        os.path.join(path, f"layer{layer_idx:02d}/chunk*_q.pt"))
    k_chunks = glob.glob(
        os.path.join(path, f"layer{layer_idx:02d}/chunk*_k.pt"))
    s_chunks = glob.glob(
        os.path.join(path, f"layer{layer_idx:02d}/chunk*_s.pt"))
    q_chunks = sorted(q_chunks)
    k_chunks = sorted(k_chunks)
    s_chunks = sorted(s_chunks)
    chunks = list(zip(q_chunks, k_chunks, s_chunks))
    random.shuffle(chunks)
    load_chunks = chunks[:num_chunks]

    query = []
    key = []
    score = []

    for chunk in load_chunks:
        query.append(torch.load(chunk[0]))
        key.append(torch.load(chunk[1]))
        score.append(torch.load(chunk[2]))

    query = torch.cat(query, dim=1).to(dst_device)
    key = torch.cat(key, dim=1).to(dst_device)
    score = torch.cat(score, dim=1).to(dst_device)
    num_items = query.shape[1]
    assert num_items == key.shape[1]
    assert num_items == score.shape[1]

    if shuffle:
        shuffle_idx = torch.randperm(num_items, device=query.device)
        query = query[:, shuffle_idx, :]
        key = key[:, shuffle_idx, :]
        score = score[:, shuffle_idx]

    return query, key, score


def train_func(args, rank):
    seed_everything(42)

    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    num_layers = (args.num_layers - args.num_skip_layers + args.mp_num -
                  1) // args.mp_num
    layer_list = []
    for i in range(num_layers):
        layer = rank * num_layers + i + args.num_skip_layers
        if layer < args.num_layers:
            layer_list.append(layer)

    os.makedirs(args.save_path, exist_ok=True)

    print(layer_list)

    for layer in layer_list:
        trainer = HashTrainer(args.num_heads,
                              args.num_kv_heads,
                              args.head_dim,
                              args.rbit,
                              args.save_path,
                              layer,
                              device=device,
                              lr=args.lr,
                              epsilon=args.epsilon,
                              lambdda=args.lambdda,
                              eta=args.eta,
                              sigma=args.sigma)
        for e in range(args.train_epochs):
            query, key, score = load_chunks(args.dataset_path, layer,
                                            args.chunk_num, device)
            for i in range(args.train_iters):
                (
                    loss,
                    similarity_loss,
                    balance_loss,
                    decorrelattion_loss,
                ) = trainer.train_for_one_step(query, key, score)
                iter = e * args.train_iters + i
                if iter % args.rep_iters == 0:
                    print(
                        f"layer {layer:2d} epoch {e:3d} iter {i:4d} sloss {similarity_loss:.5f} bloss {balance_loss:.5f} dloss {decorrelattion_loss:.5f} loss {loss:7.5f}"
                    )
            print(
                f"layer {layer:2d} epoch {e:3d} iter {i:4d} sloss {similarity_loss:.5f} bloss {balance_loss:.5f} dloss {decorrelattion_loss:.5f} loss {loss:7.5f}"
            )
        trainer.save()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    set_args(parser)
    args = parser.parse_args()
    print(args)

    work_processes = []
    for i in range(args.mp_num):
        p = mp.Process(
            target=train_func,
            args=(args, i),
        )
        p.start()
        work_processes.append(p)

    for p in work_processes:
        p.join()
