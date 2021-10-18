import torch

x = torch.tensor(data = [1, 2, 3]).float()
ll = torch.nn.Linear(3, 2)

ll.weight = torch.nn.Parameter(torch.tensor([[0, 2, 5], [1, 0, 2]]).float())
ll.bias = torch.nn.Parameter(torch.tensor([1, 1]).float())

print(f'x: {x.tolist()}')
print(f'A: {ll.weight.tolist()}')
print(f'b: {ll.bias.tolist()}')
print(f'y = Ax + b: {ll(x).tolist()}')
