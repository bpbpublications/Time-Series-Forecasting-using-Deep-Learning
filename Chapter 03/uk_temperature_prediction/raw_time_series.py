import pandas as pd
import os


def raw_time_series():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ts_df = pd.read_csv(f'{dir_path}/data/MET_Office_Weather_Data.csv')
    ts = ts_df.loc[ts_df['station'] == 'sheffield']['tmin'].tolist()
    return ts
