from statsmodels.tsa.ar_model import AutoReg
import yfinance as yf

quotes = yf.download('FB', start = '2011-1-1', end = '2021-1-1')

model = AutoReg(quotes['Close'], lags = 2)
model_fit = model.fit()

print(model_fit.params)
