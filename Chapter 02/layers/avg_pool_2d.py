import torch

A = torch.tensor([[
    [1, 2, -1, 1],
    [0, 1, -2, -1],
    [3, 0, 5, 0],
    [0, 1, 4, -3]
]]).float()

avg_pool = torch.nn.AvgPool2d(2)
out = avg_pool(A)

print(out.tolist())
