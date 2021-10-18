import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from ch5.enc_dec.model import EncoderDecoder
from ch5.enc_dec.ts import generate_ts
from ch5.training_datasets import sliding_window

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# synthetic time series length
ts_len = 2000
# lstm hidden size
hidden_size = 64
# test dataset size
test_ds_len = 200
# training epochs
epochs = 500
# input history
ts_history_len = 240
# prediction length
ts_target_len = 60

ts = generate_ts(ts_len)
X, Y = sliding_window(ts, ts_history_len, ts_target_len)
ds_len = len(X)


def to_tensor(data):
    return torch.tensor(data = data)\
        .unsqueeze(2)\
        .transpose(0, 1).float()


x_train = to_tensor(X[:ds_len - test_ds_len])
y_train = to_tensor(Y[:ds_len - test_ds_len])
x_test = to_tensor(X[ds_len - test_ds_len:])
y_test = to_tensor(Y[ds_len - test_ds_len:])

model = EncoderDecoder(hidden_size = hidden_size)
model.train()
model.train_model(x_train, y_train, epochs, ts_target_len,
                  method = 'mixed_teacher_forcing',
                  tfr = .05, lr = .005)

model.eval()
predicted = model.predict(x_test, ts_target_len)

fig, ax = plt.subplots(nrows = 3, ncols = 1)
fig.set_size_inches(7.5, 6)
for col in ax:
    r = random.randint(0, test_ds_len)
    in_seq = x_test[:, r, :].view(-1).tolist()
    target_seq = y_test[:, r, :].view(-1).tolist()
    pred_seq = predicted[:, r, :].view(-1).tolist()
    x_axis = range(len(in_seq) + len(target_seq))
    col.set_title(f'Test Sample: {r}')
    col.axis('off')
    col.plot(x_axis[:], in_seq + target_seq, color = 'blue')
    col.plot(x_axis[len(in_seq):],
             pred_seq,
             label = 'predicted',
             color = 'orange',
             linestyle = '--',
             linewidth = 3)
    col.vlines(len(in_seq), 0, 6, color = 'grey')
    col.legend(loc = "upper right")

plt.show()
