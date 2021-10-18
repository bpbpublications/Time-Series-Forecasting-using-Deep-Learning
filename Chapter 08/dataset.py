import os
import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet


def get_weather_dataset():
    encoder_length = 120
    prediction_len = 1

    dir_path = os.path.dirname(os.path.realpath(__file__))
    ts_df = pd.read_csv(f'{dir_path}/data/MET_Office_Weather_Data.csv')
    ts = ts_df.loc[ts_df['station'] == 'sheffield']['tmin'].interpolate().dropna().tolist()
    # Loading time series to pandas dataframe
    df = pd.DataFrame(
        dict(
            value = ts,
            group = [0] * len(ts),
            time_idx = np.arange(len(ts)),
        )
    )

    # Create the dataset from the pandas dataframe
    dataset = TimeSeriesDataSet(
        data = df,
        group_ids = ["group"],
        target = "value",
        time_idx = "time_idx",
        max_encoder_length = encoder_length,
        max_prediction_length = prediction_len,
        time_varying_unknown_reals = ["value"]
    )

    return dataset


if __name__ == '__main__':

    dataset = get_weather_dataset()

    print(f'Dataset size: {dataset.data["target"][0].size()[0]}')

    print('Dataset parameters:')
    print(dataset.get_parameters())

    # Convert the dataset to a dataloader
    dataloader = dataset.to_dataloader(batch_size = 8)

    # Show first 2 batches
    batch_count = 0
    for x, y in iter(dataloader):
        batch_count += 1
        if batch_count > 2:
            break
        print(f'batch: {batch_count}')
        print(f"X: {x['encoder_cont'].tolist()}")
        print(f"Y: {y[0].tolist()}")
