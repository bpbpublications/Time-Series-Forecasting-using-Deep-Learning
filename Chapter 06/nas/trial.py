import copy
import sys

import nni
import torch
from ch6.nas.model import TCN
from ch6.nas.ts import generate_time_series
from ch6.nas.training_datasets import ts_diff, get_training_datasets

# Problem Requirements
# time series input
features = 20
# training epochs
epochs = 5_00
# synthetic time series dataset
ts_len = 5_000
# test dataset size
test_len = 300

# Hyper-parameters:
p = nni.get_next_parameter()
tcl_num = p['tcl_num']
tcl_channel_size = p['tcl_channel_size']
# temporal casual layer channels
channel_sizes = [tcl_channel_size] * tcl_num
# convolution kernel size
kernel_size = p['kernel_size']
dropout = p['dropout']
slices = p['slices']
act = p['act']
use_bias = p['use_bias']

ts = generate_time_series(ts_len)

ts_diff_y = ts_diff(ts[:, 0])
ts_diff = copy.deepcopy(ts)
ts_diff[:, 0] = ts_diff_y

x_train, x_val, x_test, y_train, y_val, y_test =\
    get_training_datasets(ts_diff, features, test_len)
x_train = x_train.transpose(1, 2)
x_val = x_val.transpose(1, 2)
x_test = x_test.transpose(1, 2)
y_train = y_train[:, :, 0]
y_val = y_val[:, :, 0]
y_test = y_test[:, :, 0]

train_len = x_train.size()[0]

model_params = {
    'num_inputs':   4,
    'num_channels': channel_sizes,
    'kernel_size':  kernel_size,
    'dropout':      dropout,
    'slices':       slices,
    'act':          act,
    'use_bias':     use_bias
}
model = TCN(**model_params)

optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.005)
mse_loss = torch.nn.MSELoss()

best_params = None
min_val_loss = sys.maxsize

training_loss = []
validation_loss = []

for t in range(epochs):

    prediction = model(x_train)
    loss = mse_loss(prediction, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    val_prediction = model(x_val)
    val_loss = mse_loss(val_prediction, y_val)

    training_loss.append(loss.item())
    validation_loss.append(val_loss.item())

    if val_loss.item() < min_val_loss:
        best_params = copy.deepcopy(model.state_dict())
        min_val_loss = val_loss.item()

    if t % 10 == 0:
        diff = (y_train - prediction).view(-1).abs_().tolist()
        print(f'epoch {t}. train: {round(loss.item(), 4)}, '
              f'val: {round(val_loss.item(), 4)}')

best_model = TCN(**model_params)
best_model.eval()
best_model.load_state_dict(best_params)

tcn_prediction = best_model(x_test)

tcn_mse_loss = round(mse_loss(tcn_prediction, y_test).item(), 4)
nni.report_final_result(tcn_mse_loss)
