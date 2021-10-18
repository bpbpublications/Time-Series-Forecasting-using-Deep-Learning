import matplotlib.pyplot as plt
from ch7.stock.dataset import get_df_until_2020

q = get_df_until_2020()

q['Close'].plot()
plt.title('Microsoft')
plt.show()
