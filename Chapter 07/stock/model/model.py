import torch
import torch.nn as nn


class AlgoTrader(nn.Module):

    def __init__(self,
                 rnn_input_size,
                 ind_input_size,
                 rnn_type = 'gru',
                 rnn_hidden_size = 16,
                 ind_hidden_size = 4,
                 des_size = 4
                 ):
        super(AlgoTrader, self).__init__()
        rnn_params = {
            'input_size':  rnn_input_size,
            'hidden_size': rnn_hidden_size,
            'batch_first': True
        }
        if rnn_type == 'gru':
            self.rnn = nn.GRU(**rnn_params)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(**rnn_params)
        else:
            raise Exception(f'This type is not supported: {rnn_type}')

        self.rnn_input_size = rnn_input_size
        self.ind_input_size = ind_input_size

        self.lin_ind = nn.Linear(ind_input_size, ind_hidden_size)
        self.lin_des = nn.Linear(rnn_hidden_size + ind_hidden_size, des_size)
        self.lin_pos = nn.Linear(des_size, 1)

    def forward(self, raw, indicators, rnn_h = None):
        _, h = self.rnn(raw, rnn_h)
        z = torch.relu(self.lin_ind(indicators))
        x = torch.cat((z, h[0]), dim = 1)
        x = torch.relu(self.lin_des(x))
        p = torch.tanh(self.lin_pos(x))

        return p.view(-1)
