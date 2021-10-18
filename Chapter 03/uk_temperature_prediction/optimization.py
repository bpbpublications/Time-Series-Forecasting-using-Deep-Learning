import copy
import random
import torch
import pandas as pd
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
import sys


def get_time_series_datasets(features, test_len):
    ts_df = pd.read_csv('./data/MET_Office_Weather_Data.csv')
    ts = ts_df.loc[ts_df['station'] == 'sheffield']['tmin']
    ts = ts.interpolate().dropna().tolist()

    X = []
    Y = []

    for i in range(features + 1, len(ts) + 1):
        X.append(ts[i - (features + 1):i - 1])
        Y.append([ts[i - 1]])

    X_train, Y_train, X_test, Y_test = X[0:-test_len], Y[0:-test_len], X[-test_len:], Y[-test_len:]
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.3, shuffle = False)

    x_train = torch.tensor(data = X_train)
    y_train = torch.tensor(data = Y_train)

    x_val = torch.tensor(data = X_val)
    y_val = torch.tensor(data = Y_val)

    x_test = torch.tensor(data = X_test)
    y_test = torch.tensor(data = Y_test)

    return x_train, x_val, x_test, y_train, y_val, y_test


class DL(torch.nn.Module):

    def __init__(self, n_inp, l_1, l_2, conv1_out, conv1_kernel, conv2_kernel, drop1 = 0, drop2 = 0, n_out = 1):
        super(DL, self).__init__()
        conv1_out_ch = conv1_out
        conv2_out_ch = conv1_out * 2
        conv1_kernel = conv1_kernel
        conv2_kernel = conv2_kernel
        self.dropout_lin1 = drop1
        self.dropout_lin2 = drop2

        self.pool = torch.nn.MaxPool1d(kernel_size = 2)

        self.conv1 = torch.nn.Conv1d(in_channels = 1, out_channels = conv1_out_ch, kernel_size = conv1_kernel,
                                     padding = conv1_kernel - 1)

        self.conv2 = torch.nn.Conv1d(in_channels = conv1_out_ch, out_channels = conv2_out_ch,
                                     kernel_size = conv2_kernel,
                                     padding = conv2_kernel - 1)

        feature_tensor = self.feature_stack(torch.Tensor([[0] * n_inp]))
        self.lin1 = torch.nn.Linear(feature_tensor.size()[1], l_1)
        self.lin2 = torch.nn.Linear(l_1, l_2)
        self.lin3 = torch.nn.Linear(l_2, n_out)

    def feature_stack(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = x.flatten(start_dim = 1)
        return x

    def fc_stack(self, x):
        x1 = F.dropout(F.relu(self.lin1(x)), p = self.dropout_lin1)
        x2 = F.dropout(F.relu(self.lin2(x1)), p = self.dropout_lin2)
        y = self.lin3(x2)
        return y

    def forward(self, x):
        x = self.feature_stack(x)
        y = self.fc_stack(x)
        return y


class SarimaxPredictor(torch.nn.Module):

    def forward(self, x):
        last_values = []
        l = x.tolist()
        counter = 0
        for r in l:
            model = SARIMAX(r, order = (1, 1, 1), seasonal_order = (1, 1, 1, 12))
            results = model.fit(disp = 0)
            forecast = results.forecast()
            last_values.append([forecast[0]])
            # last_values.append([0])
            counter = counter + 1
            print(f'debug: SARIMA calculation {counter} / {len(l)}')
        return torch.tensor(data = last_values)


features = 240

x_train, x_val, x_test, y_train, y_val, y_test = get_time_series_datasets(features, 36)
loss_func = torch.nn.L1Loss()

sarimax_predictor = SarimaxPredictor()
sarimax_prediction = sarimax_predictor(x_test)
print(f'SARIMAX Loss: {loss_func(sarimax_prediction, y_test).item()}')

lin1_list = list(reversed([128, 196, 256, 320, 400, 512]))
lin2_list = list(reversed([24, 32, 48, 64, 96]))
conv_out_list = list(reversed([4, 6, 8, 12]))
conv1_kernel_list = list(reversed([6, 8, 12, 24, 36]))
conv2_kernel_list = list(reversed([8, 12, 16, 24]))
drop1_list = [0]
drop2_list = [0]

min_test_loss = 1000

for lin1 in lin1_list:
    for lin2 in lin2_list:
        for conv_out in conv_out_list:
            for conv1_kernel in conv1_kernel_list:
                for conv2_kernel in conv2_kernel_list:
                    for drop1 in drop1_list:
                        for drop2 in drop2_list:

                            random.seed(1)
                            torch.manual_seed(1)

                            print(f'testing: {lin1}-{lin2}-{conv_out}-{conv1_kernel}-{conv2_kernel}-{drop1}-{drop2}')
                            sys.stdout.flush()

                            net = DL(
                                n_inp = features,
                                l_1 = lin1,
                                l_2 = lin2,
                                conv1_out = conv_out,
                                conv1_kernel = conv1_kernel,
                                conv2_kernel = conv2_kernel,
                                drop1 = drop1,
                                drop2 = drop2,
                                n_out = 1
                            )
                            net.train()
                            optimizer = torch.optim.Adam(params = net.parameters())

                            best_model = None
                            min_val_loss = 1_000_000

                            for t in range(150):

                                prediction = net(x_train)
                                loss = loss_func(prediction, y_train)

                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()

                                val_prediction = net(x_val)
                                val_loss = loss_func(val_prediction, y_val)

                                if val_loss.item() < min_val_loss:
                                    best_model = copy.deepcopy(net)
                                    min_val_loss = val_loss.item()

                            net.eval()
                            dl_prediction = best_model(x_test)
                            test_loss = loss_func(dl_prediction, y_test).item()

                            if test_loss < min_test_loss:
                                min_test_loss = test_loss
                                print(f'best test: {test_loss} for {lin1} - {lin2} - {conv_out} - {conv1_kernel} '
                                      f'- {conv2_kernel} - {drop1} - {drop2}')
                                sys.stdout.flush()
