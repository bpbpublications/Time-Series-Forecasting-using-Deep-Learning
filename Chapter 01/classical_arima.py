from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf

from_date = '2015-1-1'
to_date = '2020-10-1'
quotes = yf.download('FB', start = from_date, end = to_date)
closes = quotes['Close'].values
train, test = closes[:-1], closes[-1]

model = ARIMA(train, order = (5, 2, 3))
results = model.fit()

forecast = results.forecast()
predicted = forecast[0]

print(f'Predicted Price on {to_date}: {round(predicted, 2)}$')
print(f'Actual Price on {to_date}: {round(test, 2)}$')
