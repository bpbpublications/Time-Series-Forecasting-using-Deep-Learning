import copy
import random
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch

from ch5.tcn.dummy import Dummy
from ch5.tcn.model import TCN
from ch5.tcn.ts import generate_time_series
from ch5.training_datasets import get_training_datasets, ts_diff, ts_int

seed = 12
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# time series input
features = 20
# training epochs
epochs = 1_000
# synthetic time series dataset
ts_len = 5_000
# test dataset size
test_len = 300
# temporal casual layer channels
channel_sizes = [10] * 4
# convolution kernel size
kernel_size = 5
dropout = .0

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
    'input_size':   4,
    'output_size':  1,
    'num_channels': channel_sizes,
    'kernel_size':  kernel_size,
    'dropout':      dropout
}
model = TCN(**model_params)

optimizer = torch.optim.Adam(params = model.parameters(), lr = .005)
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

    if t % 100 == 0:
        diff = (y_train - prediction).view(-1).abs_().tolist()
        print(f'epoch {t}. train: {round(loss.item(), 4)}, '
              f'val: {round(val_loss.item(), 4)}')

plt.title('Training Progress')
plt.yscale("log")
plt.plot(training_loss, label = 'train')
plt.plot(validation_loss, label = 'validation')
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()

best_model = TCN(**model_params)
best_model.eval()
best_model.load_state_dict(best_params)

tcn_prediction = best_model(x_test)
dummy_prediction = Dummy()(x_test)

tcn_mse_loss = round(mse_loss(tcn_prediction, y_test).item(), 4)
dummy_mse_loss = round(mse_loss(dummy_prediction, y_test).item(), 4)

plt.title(f'Test| TCN: {tcn_mse_loss}; Dummy: {dummy_mse_loss}')
plt.plot(
    ts_int(
        tcn_prediction.view(-1).tolist(),
        ts[-test_len:, 0],
        start = ts[-test_len - 1, 0]
    ),
    label = 'tcn')
plt.plot(ts[-test_len - 1:, 0], label = 'real')
plt.legend()
plt.show()
