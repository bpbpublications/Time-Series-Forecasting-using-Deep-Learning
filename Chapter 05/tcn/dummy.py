import torch.nn as nn


class Dummy(nn.Module):

    def __init__(self):
        super(Dummy, self).__init__()

    def forward(self, x):
        return x[:, -1, 0].unsqueeze(1)
