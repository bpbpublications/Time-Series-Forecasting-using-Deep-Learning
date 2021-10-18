import torch
import pandas_ta as ta

from torch import nn


def sliding_window(ts, features, target_len = 1):
    X, Y = [], []
    for i in range(features + target_len, len(ts) + 1):
        X.append(ts[i - (features + target_len):i - target_len])
        Y.append(ts[i - target_len:i])
    return X, Y


def get_indicator(q, ind_name, params):
    ts = None
    if ind_name == 'ao':
        ts = q.ta.ao(params['fast'], params['slow'])
    elif ind_name == 'apo':
        ts = q.ta.apo(params['fast'], params['slow'])
    elif ind_name == 'cci':
        ts = q.ta.cci(params['length']) / 100
    elif ind_name == 'cmo':
        ts = q.ta.cmo(params['length']) / 100
    elif ind_name == 'mom':
        ts = q.ta.mom(params['length'])
    elif ind_name == 'rsi':
        ts = q.ta.rsi(params['length']) / 100
    elif ind_name == 'tsi':
        ts = q.ta.tsi(params['fast'], params['slow']) / 100
    return ts


class NegativeMeanReturnLoss(nn.Module):

    def __init__(self):
        super(NegativeMeanReturnLoss, self).__init__()

    def forward(self, lots, price_diff):
        abs_return = torch.mul(lots.view(-1), price_diff)
        ar = torch.mean(abs_return)
        return torch.neg(ar)
