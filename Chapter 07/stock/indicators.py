import pandas_ta as ta
import matplotlib.pyplot as plt
from ch7.stock.dataset import get_df_until_2020

q = get_df_until_2020()

fig, axes = plt.subplots(nrows = 3)
fig.set_size_inches(5, 7)
axes[0].set_title('Microsoft Quotes')
q['Close'].plot(ax = axes[0])
axes[1].set_title('Momentum Indicator')
q.ta.ao(9, 14).plot(ax = axes[1])
axes[2].set_title('RSI Indicator')
q.ta.rsi(20).plot(ax = axes[2])

axes[0].xaxis.set_visible(False)
axes[1].xaxis.set_visible(False)
axes[2].xaxis.set_visible(False)
plt.show()
