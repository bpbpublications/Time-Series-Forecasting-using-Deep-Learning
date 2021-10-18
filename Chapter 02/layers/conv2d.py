import torch
from torch.nn.parameter import Parameter

A = torch.tensor([[[[1, 2, 0, 1],
                    [-1, 0, 3, 2],
                    [1, 3, 0, 1],
                    [2, -2, 1, 0]]]]).float()

conv2d = torch.nn.Conv2d(1, 1, kernel_size = 2, bias = False)
conv2d.weight = Parameter(torch.tensor([[[[1, -1], [-1, 1]]]]).float())

output = conv2d(A)

print(output)
