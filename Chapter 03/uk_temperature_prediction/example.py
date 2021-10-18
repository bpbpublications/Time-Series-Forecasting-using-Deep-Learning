import copy
import random
import sys

import torch
import matplotlib.pyplot as plt

from ch3.uk_temperature_prediction.model.dl_model import DL
from ch3.uk_temperature_prediction.model.hwes_model import HwesPredictor
from ch3.uk_temperature_prediction.model.sarima_model import SarimaxPredictor
from ch3.uk_temperature_prediction.training_datasets import get_training_datasets

random.seed(1)
torch.manual_seed(1)

features = 120

x_train, x_val, x_test, y_train, y_val, y_test =\
    get_training_datasets(features, 60)

net = DL(
    n_inp = features,
    l_1 = 400,
    l_2 = 48,
    conv1_out = 6,
    conv1_kernel = 36,
    conv2_kernel = 12,
    drop1 = .1
)
net.train()

sarima_predictor = SarimaxPredictor()
hwes_predictor = HwesPredictor()

optimizer = torch.optim.Adam(params = net.parameters())
abs_loss = torch.nn.L1Loss()

best_model = None
min_val_loss = sys.maxsize

training_loss = []
validation_loss = []

for t in range(150):

    prediction = net(x_train)
    loss = abs_loss(prediction, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    val_prediction = net(x_val)
    val_loss = abs_loss(val_prediction, y_val)

    training_loss.append(loss.item())
    validation_loss.append(val_loss.item())

    if val_loss.item() < min_val_loss:
        best_model = copy.deepcopy(net)
        min_val_loss = val_loss.item()

    if t % 10 == 0:
        print(f'epoch {t}: train - {round(loss.item(), 4)}, '
              f'val: - {round(val_loss.item(), 4)}')

best_model.eval()

dl_prediction = best_model(x_test)
sarima_prediction = sarima_predictor(x_test)
hwes_prediction = hwes_predictor(x_test)

dl_abs_loss = round(abs_loss(dl_prediction, y_test).item(), 4)
sarima_abs_loss = round(abs_loss(sarima_prediction, y_test).item(), 4)
hwes_abs_loss = round(abs_loss(hwes_prediction, y_test).item(), 4)

print('===')
print('Results on Test Dataset')
print(f'DL Loss: {dl_abs_loss}')
print(f'SARIMA Loss: {sarima_abs_loss}')
print(f'HWES Loss: {hwes_abs_loss}')

plt.title("Training progress")
plt.plot(training_loss, label = 'training loss')
plt.plot(validation_loss, label = 'validation loss')
plt.legend()
plt.show()

plt.title('Test Dataset')
plt.plot(y_test, '--', label = 'actual', linewidth = 3)
plt.plot(best_model(x_test).tolist(), label = 'DL', color = 'g')
plt.plot(sarima_prediction.tolist(), label = 'SARIMA', color = 'r')
plt.plot(hwes_prediction.tolist(), label = 'HWES', color = 'brown')
plt.legend()
plt.show()

test_n = len(y_test)
dl_abs_dev = (dl_prediction - y_test).abs_()
sarima_abs_dev = (sarima_prediction - y_test).abs_()
hwes_abs_dev = (hwes_prediction - y_test).abs_()

fig = plt.figure()

ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

ax1.set_title(f'Deep Learning Model: {dl_abs_loss}')
ax1.bar(list(range(test_n)), dl_abs_dev.view(test_n).tolist(), color = 'g')

ax2.set_title(f'SARIMA Model: {sarima_abs_loss}')
ax2.bar(list(range(test_n)), sarima_abs_dev.view(test_n).tolist(), color = 'r')

ax3.set_title(f'HWES Model: {hwes_abs_loss}')
ax3.bar(list(range(test_n)), hwes_abs_dev.view(test_n).tolist(), color = 'brown')

plt.show()
