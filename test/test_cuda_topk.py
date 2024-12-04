import torch
import myTransformer
from functools import partial
from myTransformer.cache.kernels.triton_hash_encode import hash_encode

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
data = torch.arange(128 * 10000).int().cuda()
data = data.reshape(1, 128, 10000).float()

my_output = myTransformer.capi.batch_topk(data, 1000, False)
my_sorted_indices = my_output.sort(dim=-1).values

torch_output = torch.topk(data, 1000, dim=-1, largest=False)
torch_sorted_indices = torch_output.indices.sort(dim=-1).values

print(
    (my_sorted_indices == torch_sorted_indices).sum(),
    (my_sorted_indices != torch_sorted_indices).sum(),
)

bench(partial(myTransformer.capi.batch_topk, data, 1000, False))
bench(partial(torch.topk, data, 1000, dim=-1, largest=False))
