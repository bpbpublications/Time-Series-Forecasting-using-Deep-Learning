import os

import yfinance as yf

dir_path = os.path.dirname(os.path.realpath(__file__))

full_quotes = yf.download('MSFT', start = '2009-1-1', end = '2021-1-1')
full_quotes.to_csv(f'{dir_path}/data/MSFT_until_2021_01_01.csv')

train_quotes = full_quotes[full_quotes.index < '2020-01-01']
train_quotes.to_csv(f'{dir_path}/data/MSFT_until_2020_01_01.csv')
