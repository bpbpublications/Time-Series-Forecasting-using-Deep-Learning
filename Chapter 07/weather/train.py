import copy
import os
import random
import torch
from ch7.weather.dataset import get_df_until_2016_01_01
from ch7.weather.model.model import TcnClassifier
from ch7.weather.utils import sliding_window


def prepare_model(params, save_model = False, model_name = 'tcn_rain'):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Explicitly define the end date
    end_date = '2016-01-01'
    # Historical data
    df = get_df_until_2016_01_01()
    df = df[df.index < end_date]

    # 14 days as sliding window
    w = 14
    # Number of epochs for training
    epochs = 5_00

    # Hyper-parameters:
    tcl_num = params['tcl_num']
    tcl_channel_size = params['tcl_channel_size']
    # temporal casual layer channels
    channel_sizes = [tcl_channel_size] * tcl_num
    # convolution kernel size
    kernel_size = params['kernel_size']
    dropout = params['dropout']
    slices = params['slices']
    use_bias = params['use_bias']
    lr = params['lr']

    # Australia Location for training
    locations = ['Albury', 'Newcastle', 'Richmond', 'Sydney', 'Canberra']
    # features
    features_cont = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am',
                     'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
                     'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
    features_cat = ['RainToday']

    X, Y = [], []
    for l in locations:
        df_l = df[df['Location'] == l]
        D = []
        for f in features_cont:
            D.append(df_l[f].interpolate('linear').fillna(0).values)
        for f in features_cat:
            D.append(df_l[f].map({'Yes': 1, 'No': 0}).fillna(0).values)
            # transpose to time series
        TS = []
        for i in range(df_l.shape[0]):
            row = []
            for c in D:
                row.append(c[i])
            TS.append(row)
        in_seq, out_seq = sliding_window(TS, w, 1)
        rain_seq = [r[0][-1] for r in out_seq]
        X.extend(in_seq)
        Y.extend(rain_seq)

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

    x_train = torch.tensor(X_train).float().transpose(1, 2)
    y_train = torch.tensor(Y_train).long()
    x_val = torch.tensor(X_val).float().transpose(1, 2)
    y_val = torch.tensor(Y_val).long()

    model_params = {
        'num_inputs':   len(features_cont) + len(features_cat),
        'num_classes':  2,
        'num_channels': channel_sizes,
        'kernel_size':  kernel_size,
        'dropout':      dropout,
        'slices':       slices,
        'act':          'relu',
        'use_bias':     use_bias
    }
    model = TcnClassifier(**model_params)

    optimizer = torch.optim.Adam(params = model.parameters(), lr = lr)
    cl_loss = torch.nn.CrossEntropyLoss()

    best_params = None
    min_val_loss = 1000_000

    training_loss = []
    validation_loss = []

    for t in range(epochs):

        prediction = model(x_train)
        loss = cl_loss(prediction, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_prediction = model(x_val)
        val_loss = cl_loss(val_prediction, y_val)

        training_loss.append(loss.item())
        validation_loss.append(val_loss.item())

        if val_loss.item() < min_val_loss:
            best_params = copy.deepcopy(model.state_dict())
            min_val_loss = val_loss.item()

        if t % 10 == 0:
            print(f'Epoch {t}| test: {round(loss.item(),4)}, '
                  f'val: {round(val_loss.item(),4)}')

    if save_model:
        torch.save(best_params, f'{dir_path}/data/{model_name}.pth')

    return min_val_loss
