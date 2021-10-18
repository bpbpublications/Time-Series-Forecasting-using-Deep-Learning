import matplotlib.pyplot as plt
import statsmodels.api as sm
from ch6.hybrid.training_datasets import get_aep_timeseries

if __name__ == '__main__':
    ts = get_aep_timeseries()
    ts = ts.flatten()
    hp_cycle, hp_trend = sm.tsa.filters.hpfilter(ts)
    cf_cycle, cf_trend = sm.tsa.filters.cffilter(ts)
    fig, axs = plt.subplots(5)
    fig.set_size_inches(5, 9)
    axs[0].title.set_text('Hourly Energy Consumption')
    axs[0].plot(ts[100:200])
    axs[1].title.set_text('Hodrick-Prescott: Trend filter')
    axs[1].plot(hp_trend[100:200])
    axs[2].title.set_text('Christiano Fitzgerald: Trend filter')
    axs[2].plot(cf_trend[100:200])
    axs[3].title.set_text('Hodrick-Prescott Cycle: filter')
    axs[3].plot(hp_cycle[100:200])
    axs[4].title.set_text('Christiano Fitzgerald: Cycle filter')
    axs[4].plot(cf_cycle[100:200])

    plt.show()
