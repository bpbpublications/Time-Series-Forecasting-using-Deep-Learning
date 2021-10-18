import os
import pandas as pd
import matplotlib.pyplot as plt


def interpolated_time_series():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ts_df = pd.read_csv(f'{dir_path}/data/MET_Office_Weather_Data.csv')
    ts = ts_df.loc[ts_df['station'] == 'sheffield']['tmin']\
        .interpolate().dropna().tolist()
    return ts


if __name__ == '__main__':
    ts = interpolated_time_series()
    plt.plot(ts[-120:])
    plt.show()
