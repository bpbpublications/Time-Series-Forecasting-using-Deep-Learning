import matplotlib.pyplot as plt
import yfinance as yf

stock = yf.download('FB', start = '2011-1-1', end = '2021-1-1')

close = stock['Close']
plt.xticks([])
plt.yticks([])
plt.plot(close)
plt.show()
