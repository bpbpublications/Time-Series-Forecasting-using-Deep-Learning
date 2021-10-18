import nni
import copy
import sys
import torch
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm

from ch6.hybrid.model import Hybrid
from ch6.hybrid.training_datasets import get_aep_timeseries, get_training_datasets

# PARAMETERS
# time series input length
features = 240
# length of test dataset
test_ts_len = 300

# Trial parameters:
p = nni.get_next_parameter()

trend_filter = p['trend_filter']['_name']
cycle_filter = p['cycle_filter']['_name']
use_cc = p['casual_convolution']['_name']
if use_cc:
    cc_kernel = p['casual_convolution']['kernel']
else:
    cc_kernel = None
rnn_hidden_size = p['rnn_hidden_size']
fcnn_l_num = p['fcnn_layer_num']
fcnn_l_size = p['fcnn_layer_size']

# Train parameters
learning_rate = 0.02
training_epochs = 100

# Preparing datasets for Training
ts = get_aep_timeseries()
scaler = MinMaxScaler()
scaled_ts = scaler.fit_transform(ts)
hp_cycle, hp_trend = sm.tsa.filters.hpfilter(scaled_ts)
cf_cycle, cf_trend = sm.tsa.filters.cffilter(scaled_ts)
cycle_filters = {'hp': hp_cycle, 'cf': cf_cycle}
trend_filters = {'hp': hp_trend, 'cf': cf_trend}

X = []
for i in range(len(ts)):
    row = [scaled_ts[i][0]]
    if cycle_filter != 'None':
        row.append(cycle_filters[cycle_filter][i])
    if trend_filter != 'None':
        row.append(trend_filters[trend_filter][i])
    X.append(row)

x_train, x_val, x_test, y_train, y_val, y_test =\
    get_training_datasets(X, features, test_ts_len)

y_train = y_train[:, 0].unsqueeze(1)
y_val = y_val[:, 0].unsqueeze(1)
y_test = y_test[:, 0].unsqueeze(1)

# Initializing the model
model_params = {
    'in_size':                   x_train.size(2),
    'use_casual_convolution':    use_cc,
    'casual_convolution_kernel': cc_kernel,
    'hidden_size':               rnn_hidden_size,
    'fcnn_layer_num':            fcnn_l_num,
    'fcnn_layer_size':           fcnn_l_size
}
model = Hybrid(**model_params)
model.train()

# Training
optimizer = torch.optim.Adam(params = model.parameters(), lr = learning_rate)
mse_loss = torch.nn.MSELoss()

best_params = None
min_val_loss = sys.maxsize

training_loss = []
validation_loss = []

for t in range(training_epochs):

    prediction, _ = model(x_train)
    loss = mse_loss(prediction, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    val_prediction, _ = model(x_val)
    val_loss = mse_loss(val_prediction, y_val)

    training_loss.append(loss.item())
    validation_loss.append(val_loss.item())

    if val_loss.item() < min_val_loss:
        best_params = copy.deepcopy(model.state_dict())
        min_val_loss = val_loss.item()

    if t % 10 == 0:
        print(f'epoch {t}: train - {round(loss.item(), 4)}, '
              f'val: - {round(val_loss.item(), 4)}')

best_model = Hybrid(**model_params)
best_model.eval()
best_model.load_state_dict(best_params)

_, h_list = best_model(x_val)
# warm hidden state
h = (h_list[-1, :]).unsqueeze(-2)

predicted = []
for test_seq in x_test.tolist():
    x = torch.Tensor(data = [test_seq])
    # passing hidden state through each iteration
    y, h = best_model(x, h.unsqueeze(-2))
    predicted.append(y)

acc = mse_loss(torch.tensor(predicted), y_test.view(-1)).item()
nni.report_final_result(acc)
