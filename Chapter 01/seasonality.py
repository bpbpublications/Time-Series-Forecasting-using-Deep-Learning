from math import sin
import matplotlib.pyplot as plt
import random

if __name__ == '__main__':
    random.seed(10)
    length = 100
    A = 50
    B = -.05
    C = 1
    S = 3
    trend = [A + B * i for i in range(length)]
    seasons = [S * sin(i / 5) for i in range(length)]
    noise = [C * random.gauss(0, 1) for _ in range(length)]
    ts = [trend[i] + noise[i] + seasons[i] for i in range(length)]
    plt.plot(ts)
    plt.plot(trend)
    plt.show()
