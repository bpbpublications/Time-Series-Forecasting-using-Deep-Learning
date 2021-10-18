import os
import matplotlib.pyplot as plt
import torch
from typing import OrderedDict
from ch7.stock.dataset import get_df_until_2021
from ch7.stock.model.model import AlgoTrader
from ch7.stock.utils import get_indicator, sliding_window

# Best params hyper-parameters:
params = {
    "lr":              0.01,
    "rnn_type":        "rnn",
    "rnn_hidden_size": 24,
    "ind_hidden_size": 1,
    "des_size":        2,
    "ind1":            {
        "_name":  "rsi",
        "length": 20
    },
    "ind2":            {
        "_name":  "cmo",
        "length": 20
    }
}

dir_path = os.path.dirname(os.path.realpath(__file__))
start_date = '2020-01-01'
end_date = '2021-01-01'
w = 40
q = get_df_until_2021()
q = q[q.index < end_date]
ts_len = q[q.index > start_date].shape[0]

# Preparing Sliding Window Dataset
data_source = OrderedDict()
data_source['close_diff'] = (q['Close'] - q['Close'].shift(1))
data_source['close_roc'] = (q['Close'] / q['Close'].shift(1))
data_source['ind1'] = get_indicator(q, params['ind1']['_name'], params['ind1'])
data_source['ind2'] = get_indicator(q, params['ind2']['_name'], params['ind2'])

# Cut to 'start date'
for k, v in data_source.items():
    data_source[k] = v[v.index > start_date].dropna().values

data_source['close_diff'][0] = 0
data_source['close_roc'][0] = 1

D = []
for i in range(ts_len):
    row = []
    for k, v in data_source.items():
        row.append(v[i])
    D.append(row)

X, Y = sliding_window(D, w)

# Creating tensors
x = torch.tensor(X).float()
y = torch.tensor(Y).float()

c = x[:, :, :2]
ind = x[:, -1, 2:]
tomorrow_price_diff = y[:, :, 0].view(-1)

# Initializing and loading the model
model_name = 'best_model'
model_params = {
    'rnn_input_size':  2,
    'ind_input_size':  2,
    'rnn_type':        params['rnn_type'],
    'rnn_hidden_size': params['rnn_hidden_size'],
    'ind_hidden_size': params['ind_hidden_size'],
    'des_size':        params['des_size']
}
model = AlgoTrader(**model_params)
model.load_state_dict(torch.load(f'{dir_path}/data/{model_name}.pth'))
model.eval()

with torch.no_grad():
    trades = model(c, ind)
    # Rounded Trades
    trades = torch.round(trades * 100) / 100

    # Calculating Absolute Returns
    abs_return = torch.mul(trades, tomorrow_price_diff)
    cumsum_return = [0] + torch.cumsum(abs_return, dim = 0)\
        .view(-1).tolist()
    # Buy and Hold Strategy Returns
    cumsum_price = [0] + torch.cumsum(tomorrow_price_diff, dim = 0)\
        .view(-1).tolist()

    plt.title('Trading evaluation on 2020')
    plt.plot(cumsum_return, label = 'Model Returns')
    plt.plot(cumsum_price, label = 'Buy and Hold Returns')
    plt.axhline(y = 0, color = 'black', linestyle = '--')
    plt.legend()
    plt.show()

    print(f'Model Returns: {round(cumsum_return[-1], 4)}')
    print(f'Buy and Hold Returns: {round(cumsum_price[-1], 4)}')
