import random
from math import sin, cos
import matplotlib.pyplot as plt


def get_time_series_data(length):
    a = .2
    b = 300
    c = 20
    ls = 5
    ms = 20
    gs = 100

    ts = []

    for i in range(length):
        ts.append(b + a * i + ls * sin(i / 5) + ms * cos(i / 24) + gs * sin(i / 120) + c * random.random())

    return ts


if __name__ == '__main__':
    data = get_time_series_data(3_000)
    plt.plot(data)
    plt.show()
