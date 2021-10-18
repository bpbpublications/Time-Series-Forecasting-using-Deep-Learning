import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    '/home/survex/www/kr_fnc/other_projects/ts_book/data/GlobalLandTemperatures-GlobalTemperatures.csv',
    parse_dates = [0]
)

df = df.set_index('dt')

temp = (df.resample('Y')['LandAverageTemperature'].agg(['mean']))

temp = temp[-100:]

a = temp['mean'][-1]
b = temp['mean'][0]
n = temp.shape[0]
d = (b - a) / n
trend = []

for i in range(n):
    trend.append(a + (d * i))

trend.reverse()

temp['Trend'] = trend

plt.title('Global Temperature Average')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.plot(temp)
plt.show()
