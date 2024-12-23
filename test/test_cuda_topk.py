import torch
import myTransformer
from functools import partial
from myTransformer.cache.kernels.triton_hash_encode import hash_encode
from collections import Counter

torch.cuda.set_device(6)
torch.manual_seed(42)


def bench(func):
    import time
    import numpy as np

    for i in range(5):
        func()

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(100):
        func()
    torch.cuda.synchronize()
    t1 = time.time()
    print((t1 - t0) * 1000 / 100)


# data = torch.randn(1, 128, 10000).cuda() * 10
# data = data.int()
# data = torch.arange(128 * 10000).int().cuda()
# data = data.reshape(1, 128, 10000).float()
BSZ = 2
SEQ = 96000
SEL = 3840
NUM_HEAD = 32
data = torch.randn(BSZ, NUM_HEAD, SEQ, dtype=torch.float16,
                   device='cuda')  #  * 1000
# data = data.abs().to(torch.int32).to(torch.float16)

my_output = myTransformer.capi.batch_topk(data, SEL, False)
my_sorted_indices = my_output.sort(dim=-1).values
# print(my_sorted_indices)

torch_output = torch.topk(data, SEL, dim=-1, largest=False)
torch_sorted_indices = torch_output.indices.sort(dim=-1).values
# print(torch_sorted_indices)

diff = 0
for i in range(NUM_HEAD):
    t = torch_sorted_indices[:, i, :].tolist()[0]
    m = my_sorted_indices[:, i, :].tolist()[0]
    # print(t)
    # print(m)
    c = Counter(t + m)
    diff = diff + 2 * len(c) - len(t) - len(m)

print(f"diff: {diff} / total: {SEL * NUM_HEAD}")

torch.cuda.nvtx.range_push("batch_topk")
bench(partial(myTransformer.capi.batch_topk, data, SEL, False))
torch.cuda.nvtx.range_pop()
bench(partial(torch.topk, data, SEL, dim=-1, largest=False))
