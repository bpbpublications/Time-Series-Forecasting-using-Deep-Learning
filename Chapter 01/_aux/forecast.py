import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
import yfinance as yf

symbol = 'BTC-USD'
q: pd.DataFrame = yf.download(symbol, start = '2017-1-1', end = '2017-11-1')


def arima_model(timeseries):
    automodel = pm.auto_arima(timeseries,
                              start_p = 1,
                              start_q = 1,
                              test = "adf",
                              seasonal = False,
                              trace = True)
    return automodel


def plot_arima(n_periods, timeseries, automodel: pm.ARIMA, alpha = .05):
    # Прогноз с доверительным интервалом
    fc, confint = automodel.predict(n_periods = n_periods, return_conf_int = True, alpha = alpha)

    fc_series = pd.Series(fc)
    lower_series = pd.Series(confint[:, 0])
    upper_series = pd.Series(confint[:, 1])

    # Смещаем вправо
    fc_series.index = fc_series.index + len(timeseries)
    lower_series.index = lower_series.index + len(timeseries)
    upper_series.index = upper_series.index + len(timeseries)

    plt.figure(figsize = (10, 6))
    plt.plot(timeseries)
    plt.plot(fc_series, color = "red")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    plt.fill_between(lower_series.index,
                     lower_series,
                     upper_series,
                     color = "k", alpha = .25)
    confid_perc = (1 - alpha) * 100
    plt.legend(("past", "forecast", f"{confid_perc}% confidence interval"), loc = "upper left")
    plt.show()


closes = [v for _, v in q['Close'].iteritems()]

automodel: pm.ARIMA = arima_model(closes)

# print(automodel.summary())

plot_arima(100, closes, automodel, 0.3)
