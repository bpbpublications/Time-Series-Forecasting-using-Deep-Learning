import os
from datetime import datetime

import pandas as pd


def get_df_until_2021_02_01():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(
        f'{dir_path}/data/COVID_19_until_2021_02_01.csv',
        date_parser = lambda d: datetime.strptime(d, '%Y-%m-%d'),
        index_col = 'Date'
    )
    return df


def get_df_complete():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(
        f'{dir_path}/data/COVID_19_complete.csv',
        date_parser = lambda d: datetime.strptime(d, '%Y-%m-%d'),
        index_col = 'Date'
    )
    return df
