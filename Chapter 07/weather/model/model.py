import torch
from typing import OrderedDict

import torch.nn as nn
from torch.nn.utils import weight_norm


class Crop(nn.Module):

    def __init__(self, crop_size):
        super(Crop, self).__init__()
        self.crop_size = crop_size

    def forward(self, x):
        return x[:, :, :-self.crop_size].contiguous()


class TemporalCasualLayer(nn.Module):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 kernel_size,
                 stride,
                 dilation,
                 dropout = 0.2,
                 act = 'relu',
                 slices = 2,
                 use_bias = True
                 ):
        super(TemporalCasualLayer, self).__init__()
        padding = (kernel_size - 1) * dilation
        conv_params = {
            'kernel_size': kernel_size,
            'stride':      stride,
            'padding':     padding,
            'dilation':    dilation
        }
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh()
        }

        self.use_bias = use_bias

        layers = OrderedDict()
        for s in range(1, slices + 1):
            if s == 1:
                layers[f'conv{s}'] = weight_norm(nn.Conv1d(n_inputs, n_outputs, **conv_params))
            else:
                layers[f'conv{s}'] = weight_norm(nn.Conv1d(n_outputs, n_outputs, **conv_params))
            layers[f'crop{s}'] = Crop(padding)
            layers[f'act{s}'] = activations[act]
            layers[f'dropout{s}'] = nn.Dropout(dropout)

        self.net = nn.Sequential(layers)

        if n_inputs != n_outputs and use_bias:
            self.bias = nn.Conv1d(n_inputs, n_outputs, 1)
        else:
            self.bias = None

        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.net(x)
        if self.use_bias:
            b = x if self.bias is None else self.bias(x)
            return self.relu(y + b)
        else:
            return self.relu(y)


class TemporalConvolutionNetwork(nn.Module):

    def __init__(self,
                 num_inputs,
                 num_channels,
                 kernel_size = 2,
                 dropout = 0.2,
                 slices = 2,
                 act = 'relu',
                 use_bias = True
                 ):
        super(TemporalConvolutionNetwork, self).__init__()
        layers = []
        num_levels = len(num_channels)

        tcl_param = {
            'kernel_size': kernel_size,
            'stride':      1,
            'dropout':     dropout,
            'slices':      slices,
            'act':         act,
            'use_bias':    use_bias
        }

        for i in range(num_levels):
            dilation = 2**i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            tcl_param['dilation'] = dilation
            tcl = TemporalCasualLayer(in_ch, out_ch, **tcl_param)
            layers.append(tcl)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TcnClassifier(nn.Module):

    def __init__(self, **params):
        super(TcnClassifier, self).__init__()
        self.num_channels = params['num_channels']
        self.num_classes = params.pop('num_classes')

        self.tcn = TemporalConvolutionNetwork(**params)
        self.linear = nn.Linear(self.num_channels[-1], self.num_classes)

    def forward(self, x):
        x = self.tcn(x)
        x = self.linear(x[:, :, -1])
        y = torch.log_softmax(x, dim = 1)
        return y
