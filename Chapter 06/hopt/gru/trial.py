import copy
import sys
import nni
import torch
from sklearn.preprocessing import MinMaxScaler
from ch6.hopt.gru.model.gru import GRU
from ch6.hopt.gru.training_datasets import get_pjme_timeseries, get_training_datasets

# Problem Requirements:
# length of sliding window
features = 240
# length of test dataset
test_ts_len = 100

#Trial params:
trial_params = nni.get_next_parameter()

# optimizer name
optimizer_name = trial_params['optimizer']
# size of GRU hidden state
gru_hidden_size = trial_params['gru_hidden_size']
# Optimizer learning rate
learning_rate = trial_params['learning_rate']

training_epochs = 50

# Preparing datasets for Training
ts = get_pjme_timeseries()
scaler = MinMaxScaler()
scaled_ts = scaler.fit_transform(ts)
x_train, x_val, x_test, y_train, y_val, y_test =\
    get_training_datasets(scaled_ts, features, test_ts_len)

# Initializing the model
model = GRU(hidden_size = gru_hidden_size)
model.train()

# Training
optimizers = {
    'adam':   torch.optim.Adam,
    'sgd':    torch.optim.SGD,
    'adamax': torch.optim.Adamax
}

optimizer = optimizers[optimizer_name](params = model.parameters(), lr = learning_rate)
mse_loss = torch.nn.MSELoss()

best_model = None
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
        best_model = copy.deepcopy(model)
        min_val_loss = val_loss.item()

best_model.eval()
_, h_list = best_model(x_val)
# warm hidden state
h = (h_list[-1, :]).unsqueeze(-2)

predicted = []
for test_seq in x_test.tolist():
    x = torch.Tensor(data = [test_seq])
    # passing hidden state through each iteration
    y, h = best_model(x, h.unsqueeze(-2))
    predicted.append(y)

test_loss = mse_loss(
    torch.tensor(predicted),
    y_test.view(-1)).item()

nni.report_final_result(test_loss)
