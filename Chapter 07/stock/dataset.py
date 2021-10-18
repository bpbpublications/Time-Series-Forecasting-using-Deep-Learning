import os
from datetime import datetime
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_df_until_2020():
    df = pd.read_csv(
        f'{dir_path}/data/MSFT_until_2020_01_01.csv',
        date_parser = lambda d: datetime.strptime(d, '%Y-%m-%d'),
        index_col = 'Date'
    )
    return df


def get_df_until_2021():
    df = pd.read_csv(
        f'{dir_path}/data/MSFT_until_2021_01_01.csv',
        date_parser = lambda d: datetime.strptime(d, '%Y-%m-%d'),
        index_col = 'Date'
    )
    return df


if __name__ == '__main__':
    q = get_df_until_2021()
    print(f'Indicator list: {q.ta.indicators()}')
