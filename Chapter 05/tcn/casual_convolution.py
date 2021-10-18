import torch
from torch.nn import Parameter

x = torch.tensor([[[0, 1, 2, 3, 4]]]).float()

k = 3
conv1d = torch.nn.Conv1d(1, 1, kernel_size = k, padding = k - 1, bias = False)
conv1d.weight = Parameter(torch.tensor([[[1, 0, -1]]]).float())

y1 = conv1d(x)
y2 = y1[:, :, :-(k - 1)]

print(y2.tolist())
