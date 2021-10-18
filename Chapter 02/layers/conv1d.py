import torch
from torch.nn.parameter import Parameter

A = torch.tensor([[[1, 0, 2, 0, 3, 0]]]).float()

conv1d = torch.nn.Conv1d(1, out_channels = 2, kernel_size = 3, bias = False)
conv1d.weight = Parameter(torch.tensor([[[1, 0, -1]], [[0, 2, 0]]]).float())

output = conv1d(A)

print(output)
