from statsmodels.tsa.statespace.sarimax import SARIMAX
import yfinance as yf

from_date = '2015-1-1'
to_date = '2020-10-1'
quotes = yf.download('FB', start = from_date, end = to_date)
closes = quotes['Close'].values
train, test = closes[:-1], closes[-1]

model = SARIMAX(train, order = (3, 1, 1), seasonal_order = (0, 0, 0, 0))
results = model.fit(disp = 0)

forecast = results.forecast()
predicted = forecast[0]

print(f'Predicted Price on {to_date}: {round(predicted, 2)}$')
print(f'Actual Price on {to_date}: {round(test, 2)}$')
