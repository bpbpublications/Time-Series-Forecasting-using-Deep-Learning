import random
from math import sin
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


def detrend(ts):
    X = [[i] for i in range(len(ts))]
    y = np.array(ts).reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    a = reg.coef_[0][0]
    b = reg.intercept_[0]
    detrend_ts = [(ts[i] - a * i - b) for i in range(len(ts))]
    return detrend_ts, a, b


def retrend(ts, a, b):
    return [(ts[i] + a * i + b) for i in range(len(ts))]


if __name__ == '__main__':

    random.seed(1)

    ts = [10 + .8 * i + sin(i) + 3 * random.random() for i in range(20)]
    detrend_ts, a, b = detrend(ts)
    retrend_ts = retrend(detrend_ts, a, b)

    fig = plt.figure()

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    ax1.set_title("Trended Time Series")
    ax1.plot(ts)

    ax2.set_title("Detrended Time Series")
    ax2.plot(detrend_ts)

    ax3.set_title("Retrended Time Series")
    ax3.plot(retrend_ts)

    plt.show()
