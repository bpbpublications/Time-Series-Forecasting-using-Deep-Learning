import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pytorch_forecasting import TimeSeriesDataSet, DeepAR
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import pytorch_lightning as pl
from ch8.forecasting_model import ForecastingModel

random.seed(1)
torch.manual_seed(1)

# define dataset
encode_length = 120
prediction_length = 1
training_cutoff = 2015  # year for cutoff

dir_path = os.path.dirname(os.path.realpath(__file__))
ts_df = pd.read_csv(f'{dir_path}/data/MET_Office_Weather_Data.csv')
ts_df = ts_df.loc[(ts_df['station'] == 'sheffield')]
train_ts = ts_df.loc[ts_df['year'] < training_cutoff]['tmin']\
    .interpolate().dropna().tolist()

test_ts = ts_df.loc[ts_df['year'] >= training_cutoff - (encode_length / 12)]['tmin']\
    .interpolate().dropna().tolist()

train_data = pd.DataFrame(
    dict(
        value = train_ts,
        group = [0] * len(train_ts),
        time_idx = np.arange(len(train_ts)),
    )
)

test_data = pd.DataFrame(
    dict(
        value = test_ts,
        group = [0] * len(test_ts),
        time_idx = np.arange(len(test_ts)),
    )
)

time_series_dataset_params = {
    'group_ids':                  ["group"],
    'target':                     "value",
    'time_idx':                   "time_idx",
    'max_encoder_length':         encode_length,
    'max_prediction_length':      prediction_length,
    'time_varying_unknown_reals': ["value"]
}

# create the dataset from the pandas dataframe
testing = TimeSeriesDataSet(data = test_data, **time_series_dataset_params)
training = TimeSeriesDataSet(data = train_data, **time_series_dataset_params)
validation = TimeSeriesDataSet.from_dataset(
    training,
    train_data,
    min_prediction_idx = training.index.time.max() + 1,
    stop_randomization = True
)

bs = 240

train_dataloader = training.to_dataloader(train = True, batch_size = bs)
val_dataloader = validation.to_dataloader(train = False, batch_size = bs)
test_dataloader = testing.to_dataloader(train = False, batch_size = bs)

# define trainer with early stopping
early_stop_callback = EarlyStopping(
    monitor = "val_loss",
    min_delta = 1e-5,
    patience = 1,
    verbose = False,
    mode = "min"
)

lr_logger = LearningRateMonitor()

trainer = pl.Trainer(
    max_epochs = 1000,
    gpus = 0,
    gradient_clip_val = 0.1,
    limit_train_batches = 30,
    callbacks = [lr_logger, early_stop_callback],
)

custom_model = ForecastingModel.from_dataset(
    dataset = training,
    l_1 = 400,
    l_2 = 48,
    conv1_out = 6,
    conv1_kernel = 36,
    conv2_kernel = 12,
    drop1 = .1
)
deepar_model = DeepAR.from_dataset(dataset = training)

models = {
    'Custom':              custom_model,
    'Deep Autoregressive': deepar_model
}

predictions = {}
for model_name, model in models.items():

    # find optimal learning rate
    res = trainer.tuner.lr_find(
        custom_model,
        train_dataloader = train_dataloader, val_dataloaders = val_dataloader,
        early_stop_threshold = 1000.0,
        max_lr = 0.3,
    )

    print(f"Suggested Learning Rate for {model_name}: {res.suggestion()}")

    trainer.fit(
        custom_model,
        train_dataloader = train_dataloader, val_dataloaders = val_dataloader
    )

    predictions[model_name] = custom_model.predict(test_dataloader)

x, y = next(iter(test_dataloader))
plt.title('Predictions on Test dataset')
plt.plot(y[0].tolist(), label = 'Real')
for model_name, prediction in predictions.items():
    plt.plot(prediction.tolist(), label = model_name)
plt.legend()
plt.show()