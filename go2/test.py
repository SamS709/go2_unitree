import torch
import sys
torch.set_printoptions(precision=2, threshold=sys.maxsize, linewidth=200, edgeitems=100)

t = torch.zeros((2, 5, 3), dtype = torch.float32)

print(t)