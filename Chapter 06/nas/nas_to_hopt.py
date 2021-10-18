from typing import OrderedDict
import torch.nn as nn

class FCNN(nn.Module):

    def __init__(self, hidden_layers_num):
        super(FCNN, self).__init__()
        assert hidden_layers_num >= 0, 'hidden layers number should be positive or zero'
        self.lin_first = nn.Linear(5, 10)
        hidden_layers = OrderedDict()
        for l in range(hidden_layers_num):
            hidden_layers[f'lin_hidden{l}'] = nn.Linear(10, 10)
        self.lin_hidden = nn.Sequential(hidden_layers)
        self.lin_last = nn.Linear(10, 1)
