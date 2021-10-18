import os
import random
import torch
from sklearn.metrics import balanced_accuracy_score
from ch7.weather.dataset import get_df_complete
from ch7.weather.model.model import TcnClassifier
from ch7.weather.utils import sliding_window

dir_path = os.path.dirname(os.path.realpath(__file__))

location = 'Sydney'
w = 14
# We pick 14 before the 2016-01-01 for the sliding window
from_date = '2015-12-17'
to_date = '2017-01-01'

date_fmt = '%Y-%m-%d'
df = get_df_complete()

# features
features_cont = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am',
                 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
                 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
features_cat = ['RainToday']

X, Y = [], []
df_l = df[(df['Location'] == location) & (df.index < to_date) & (df.index > from_date)]
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

x_test = torch.tensor(X).float().transpose(1, 2)

model_name = 'best_model'

model_params = {
    'num_inputs':   len(features_cont) + len(features_cat),
    'num_classes':  2,
    'num_channels': [32] * 2,
    'act':          'relu',
    "kernel_size":  7,
    "dropout":      0.1,
    "slices":       1,
    "use_bias":     True
}

model = TcnClassifier(**model_params)
model.load_state_dict(torch.load(f'{dir_path}/data/{model_name}.pth'))
model.eval()

predicted_prob = model(x_test)
model_prediction = predicted_prob.data.max(1, keepdim = True)[1].view(-1).tolist()
no_rain_prediction = [0] * len(Y)
no_sun_prediction = [1] * len(Y)
coin_flip_prediction = [random.randint(0, 1) for _ in range(len(Y))]
tomorrow_like_today_prediction = x_test[:, -1, -1].view(-1).tolist()

ba_sc = balanced_accuracy_score
model_score = round(ba_sc(Y, model_prediction), 4)
no_rain_score = round(ba_sc(Y, no_rain_prediction), 4)
no_sun_score = round(ba_sc(Y, no_sun_prediction), 4)
coin_flip_score = round(ba_sc(Y, coin_flip_prediction), 4)
tlt_score = round(ba_sc(Y, tomorrow_like_today_prediction), 4)

print(f'Model Prediction Score: {model_score}')
print(f'No Rain Prediction Score: {no_rain_score}')
print(f'Rain Prediction Score: {no_sun_score}')
print(f'Coin Flip Prediction Score: {coin_flip_score}')
print(f'Tomorrow like Today Prediction Score: {tlt_score}')
