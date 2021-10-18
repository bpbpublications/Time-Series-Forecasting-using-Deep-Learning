import copy
import os
import torch
from typing import OrderedDict
from torch import optim
from ch7.stock.dataset import get_df_until_2020
from ch7.stock.model.model import AlgoTrader
from ch7.stock.utils import sliding_window, NegativeMeanReturnLoss, get_indicator


def prepare_model(params, save_model = False, model_name = 'algo_trader'):
    # Global Parameters
    dir_path = os.path.dirname(os.path.realpath(__file__))

    start_date = '2010-01-01'
    end_date = '2020-01-01'
    w = 40
    epochs = 5_00
    train_val_ratio = .8

    #Hyper parameters
    lr = params['lr']
    rnn_type = params['rnn_type']
    rnn_hidden_size = params['rnn_hidden_size']
    ind_hidden_size = params['ind_hidden_size']
    des_size = params['des_size']
    ind1_name = params['ind1']['_name']
    ind2_name = params['ind2']['_name']

    # Preparing Sliding Window Dataset
    q = get_df_until_2020()
    q = q[q.index < end_date]
    ts_len = q[q.index > start_date].shape[0]
    train_len = int(ts_len * train_val_ratio)

    data_source = OrderedDict()
    data_source['close_diff'] = (q['Close'] - q['Close'].shift(1))
    data_source['close_roc'] = (q['Close'] / q['Close'].shift(1))
    data_source['ind1'] = get_indicator(q, ind1_name, params['ind1'])
    data_source['ind2'] = get_indicator(q, ind2_name, params['ind2'])

    # Cut to 'start date'
    for k, v in data_source.items():
        data_source[k] = v[v.index > start_date].dropna().values

    D = []
    for i in range(ts_len):
        row = []
        for k, v in data_source.items():
            row.append(v[i])
        D.append(row)

    X, Y = sliding_window(D, w)

    # Train / Validation split
    X_test, Y_test = X[:train_len], Y[:train_len]
    X_val, Y_val = X[train_len:], Y[train_len:]

    # Preparing tensors
    x_test = torch.tensor(X_test).float()
    y_test = torch.tensor(Y_test).float()
    x_val = torch.tensor(X_val).float()
    y_val = torch.tensor(Y_val).float()

    c_test, c_val = x_test[:, :, :2], x_val[:, :, :2]
    ind_test, ind_val = x_test[:, -1, 2:], x_val[:, -1, 2:]
    p_test, p_val = y_test[:, :, 0].view(-1), y_val[:, :, 0].view(-1)

    # Model Initializing
    model_params = {
        'rnn_input_size':  2,
        'ind_input_size':  2,
        'rnn_type':        rnn_type,
        'rnn_hidden_size': rnn_hidden_size,
        'ind_hidden_size': ind_hidden_size,
        'des_size':        des_size
    }
    model = AlgoTrader(**model_params)

    # Training
    optimizer = optim.Adam(model.parameters(), lr = lr)
    criterion = NegativeMeanReturnLoss()

    min_val_loss = 1000_000
    best_params = None

    for e in range(epochs):
        predicted = model(c_test, ind_test)
        loss = criterion(predicted, p_test)
        optimizer.zero_grad()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_predicted = model(c_val, ind_val)
        val_loss = criterion(val_predicted, p_val)

        if val_loss.item() < min_val_loss:
            min_val_loss = val_loss.item()
            best_params = copy.deepcopy(model.state_dict())

        if e % 10 == 0:
            print(f'Epoch {e}| test:{round(loss.item(), 4)},'
                  f'val: {round(val_loss.item(), 4)}')

    # Saving the best model if necessary
    if save_model:
        torch.save(best_params, f'{dir_path}/data/{model_name}.pth')

    return min_val_loss
