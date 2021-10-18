import matplotlib.pyplot as plt
from ch7.stock.dataset import get_df_until_2020

q = get_df_until_2020()

(q['Close'] - q['Close'].shift(1)).plot()
plt.title('Microsoft Quotes Absolute Change')
plt.show()

(q['Close'] / q['Close'].shift(1)).plot()
plt.title('Microsoft Quotes Relative Change')
plt.show()
