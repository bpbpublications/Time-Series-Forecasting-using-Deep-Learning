import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import datetime as dt
import statsmodels.api as sm

from ch7.covid.dataset import get_df_until_2021_02_01

df = get_df_until_2021_02_01()
date_fmt = '%Y-%m-%d'
start_date = datetime.datetime.strptime('2020-01-22', date_fmt)
end_date = datetime.datetime.strptime('2021-02-01', date_fmt)
date_list = mdates.drange(start_date, end_date, dt.timedelta(days = 1))

# Total confirmed cases
aus_df = df[df['Country'] == 'Austria']
aus_confirmed = aus_df['Confirmed'].values
plt.title('Total number of Confirmed cases')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval = 90))
plt.plot(date_list[:], aus_confirmed, color = 'blue')
plt.show()

# Daily increase
aus_confirmed_diff = df[df['Country'] == 'Austria']['Confirmed'].diff().fillna(0)
plt.title('Daily Increase of Confirmed cases')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval = 90))
plt.plot(date_list[:], aus_confirmed_diff, color = 'blue')
plt.show()

# Daily increase with neighbours
countries = ['Italy', 'Russia', 'Hungary', 'Austria', 'Israel', 'Poland']

plt.title('Daily Increase of Confirmed cases by Countries')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval = 90))
for c in countries:
    plt.title(f'Confirmed')
    ts = df[df['Country'] == c]['Confirmed'].diff().fillna(0).values
    ts = ts / max(ts)
    plt.plot(ts, label = c)
plt.legend()
plt.show()

# Additional statistical tools
aus_confirmed_diff_norm = aus_confirmed_diff / max(aus_confirmed_diff)
_, train_hp_trend = sm.tsa.filters.hpfilter(aus_confirmed_diff_norm)
train_cf_cycle, _ = sm.tsa.filters.cffilter(aus_confirmed_diff_norm)
plt.title('Trend and Cycle filters')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval = 90))
plt.plot(date_list[:], aus_confirmed_diff_norm, color = 'blue', label = 'original')
plt.plot(date_list[:], train_hp_trend, color = 'orange', label = 'trend filter', linewidth = 3)
plt.plot(date_list[:], train_cf_cycle, color = 'green', label = 'cycle filter')
plt.legend()
plt.show()
