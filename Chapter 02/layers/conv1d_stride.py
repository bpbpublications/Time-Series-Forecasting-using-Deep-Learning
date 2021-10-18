import torch
from torch.nn.parameter import Parameter

A = torch.tensor([[[1, 2, 3, 4, 5]]]).float()

conv1d = torch.nn.Conv1d(1, 1, kernel_size = 3, bias = False, stride = 2)
conv1d.weight = Parameter(torch.tensor([[[1, 0, -1]]]).float())

output = conv1d(A)
print(output.tolist())
