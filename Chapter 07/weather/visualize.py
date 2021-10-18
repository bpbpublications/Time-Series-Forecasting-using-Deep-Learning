import matplotlib.pyplot as plt
from ch7.weather.dataset import get_df_until_2016_01_01

df = get_df_until_2016_01_01()

df_syd = df[(df['Location'] == 'Sydney') &
            (df.index > '2015-01-01')]

rainfall = df_syd['Rainfall'].values
raintoday = df_syd['RainToday']\
    .map({'Yes': 1, 'No': 0}).values

fig, axs = plt.subplots(2)
axs[0].set_title('Rainfall')
axs[0].plot(rainfall)
axs[0].axis('off')
axs[1].set_title('Rain Classification')
axs[1].bar(range(len(raintoday)), raintoday, color = 'red')
axs[1].axis('off')
plt.show()
