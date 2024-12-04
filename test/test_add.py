import torch
import myTransformer

a = torch.arange(0, 100).float().cuda()
b = torch.arange(100, 0, -1).float().cuda()
c = torch.zeros(100).float().cuda()

print(myTransformer.capi.add(a, b, c))
