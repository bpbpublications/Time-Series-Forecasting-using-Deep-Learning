import torch

A = torch.tensor([[
    [1, 2, -1, 1],
    [0, 1, -2, -1],
    [3, 0, 5, 0],
    [0, 1, 4, -3]
]]).float()

max_pool = torch.nn.MaxPool2d(2)
out = max_pool(A)

print(out.tolist())
