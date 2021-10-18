import random
from math import sin, cos
import matplotlib.pyplot as plt


def normalize(ts):
    max_ts = max(ts)
    min_ts = min(ts)
    normal_ts = [(v - min_ts) / (max_ts - min_ts) for v in ts]
    return normal_ts, max_ts, min_ts


def denormalize(ts, max_ts, min_ts):
    denormal_ts = [v * (max_ts - min_ts) + min_ts for v in ts]
    return denormal_ts


if __name__ == '__main__':

    random.seed(1)

    ts = [10 * sin(i) * cos(i) * cos(i) for i in range(20)]
    normal_ts, max_ts, min_ts = normalize(ts)
    denormal_ts = denormalize(normal_ts, max_ts, min_ts)

    fig = plt.figure()

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    ax1.set_title("Raw Time Series")
    ax1.plot(ts)

    ax2.set_title("Normalized Time Series")
    ax2.plot(normal_ts)

    ax3.set_title("Denormalized Time Series")
    ax3.plot(denormal_ts)

    plt.show()
