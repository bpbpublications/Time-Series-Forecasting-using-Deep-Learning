import os
from datetime import datetime
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_df_until_2016_01_01():
    df = pd.read_csv(
        f'{dir_path}/data/weatherAUS_until_2016_01_01.csv',
        date_parser = lambda d: datetime.strptime(d, '%Y-%m-%d'),
        index_col = 'Date'
    )
    return df


def get_df_complete():
    df = pd.read_csv(
        f'{dir_path}/data/weatherAUS_complete.csv',
        date_parser = lambda d: datetime.strptime(d, '%Y-%m-%d'),
        index_col = 'Date'
    )
    return df
