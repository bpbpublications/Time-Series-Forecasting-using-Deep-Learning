import torch
from typing import OrderedDict

import torch.nn as nn
from torch.nn.utils import weight_norm


class CasualConvolution(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size = 5):
        super(CasualConvolution, self).__init__()
        self.kernel_size = kernel_size
        self.conv = weight_norm(
            nn.Conv1d(n_inputs, n_outputs,
                      kernel_size = kernel_size,
                      padding = kernel_size - 1))

    def forward(self, x):
        x1 = torch.swapaxes(x, 1, 2)
        x1 = self.conv(x1)
        x1 = x1[:, :, :-(self.kernel_size - 1)].contiguous()
        x = torch.swapaxes(x1, 2, 1)
        return x


class Hybrid(nn.Module):

    def __init__(self,
                 hidden_size,
                 in_size = 1,
                 out_size = 1,
                 use_casual_convolution = False,
                 casual_convolution_kernel = 5,
                 fcnn_layer_num = 0,
                 fcnn_layer_size = 8
                 ):
        super(Hybrid, self).__init__()

        if use_casual_convolution:
            self.cc = CasualConvolution(in_size, in_size, casual_convolution_kernel)
        else:
            self.cc = None

        self.rnn = nn.RNN(
            input_size = in_size,
            hidden_size = hidden_size,
            batch_first = True)

        lin_layers = OrderedDict()

        for l in range(fcnn_layer_num):
            if l == 0:
                lin_layers[f'lin_hidden_{l}'] = nn.Linear(hidden_size, fcnn_layer_size)
            else:
                lin_layers[f'lin_hidden_{l}'] = nn.Linear(fcnn_layer_size, fcnn_layer_size)

        if fcnn_layer_num == 0:
            lin_layers['lin_final'] = nn.Linear(hidden_size, out_size)
        else:
            lin_layers['lin_final'] = nn.Linear(fcnn_layer_size, out_size)

        self.fc = nn.Sequential(lin_layers)

    def forward(self, x, h = None):

        if self.cc:
            x = self.cc(x)

        out, _ = self.rnn(x, h)
        last_hidden_states = out[:, -1]
        out = self.fc(last_hidden_states)
        return out, last_hidden_states
