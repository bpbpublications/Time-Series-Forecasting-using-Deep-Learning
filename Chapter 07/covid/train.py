import os
import random
import torch
import statsmodels.api as sm
from ch7.covid.dataset import get_df_until_2021_02_01
from ch7.covid.model.model import EncoderDecoder
from ch7.covid.utils import sliding_window


def prepare_model(params, save_model = False, model_name = 'enc_dec'):
    # Global Parameters
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = get_df_until_2021_02_01()
    start_date = '2020-01-01'
    end_date = '2021-02-01'
    countries = ['Italy', 'Russia', 'Hungary', 'Austria', 'Israel', 'Poland']

    # size of the sliding window
    w = 120
    # length of prediction
    out = 60
    # number of training epochs
    epochs = 2_00

    # Hyper-parameters:
    hidden_size = params['hidden_size']
    hidden_dl_size = params['hidden_dl_size']
    lr = params['lr']
    tfr = params['tfr']

    # Preparing Sliding-window Datasets
    X, Y = [], []
    for c in countries:
        # diff
        ts_df = df[df['Country'] == c]['Confirmed'].diff().dropna()
        train = ts_df[start_date:end_date].values
        # Normalized time series
        train = train / max(train)
        # Statistical pre-processing
        _, train_hp_trend = sm.tsa.filters.hpfilter(train)
        train_cf_cycle, _ = sm.tsa.filters.cffilter(train)

        D = []
        for i in range(len(train)):
            D.append([train[i], train_hp_trend[i], train_cf_cycle[i]])

        # input - output for country
        X_c, Y_c = sliding_window(D, w, out)
        X.extend(X_c)
        Y.extend(Y_c)

    # Train-Validation Split
    X_train, Y_train = [], []
    X_val, Y_val = [], []
    for i in range(len(X)):
        if random.random() > .8:
            X_val.append(X[i])
            Y_val.append(Y[i])
        else:
            X_train.append(X[i])
            Y_train.append(Y[i])

    # Converting datasets to tensors
    x_train = torch.tensor(X_train).float().transpose(0, 1)
    y_train = torch.tensor(Y_train).float().transpose(0, 1)[:, :, 0]
    x_val = torch.tensor(X_val).float().transpose(0, 1)
    y_val = torch.tensor(Y_val).float().transpose(0, 1)[:, :, 0]

    # Initializing the model
    model_params = {
        'hidden_size':    hidden_size,
        'hidden_dl_size': hidden_dl_size,
        'input_size':     3,
        'output_size':    1
    }
    model = EncoderDecoder(**model_params)
    model.train()

    # Training and getting the results
    model_params, val = model.train_model(
        x_train, y_train, x_val, y_val, epochs, out,
        method = 'mixed_teacher_forcing', tfr = tfr, lr = lr)

    # Saving the model if necessary
    if save_model:
        torch.save(model_params, f'{dir_path}/data/{model_name}.pth')

    return val
