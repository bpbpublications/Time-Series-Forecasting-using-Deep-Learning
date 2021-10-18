from math import sin
import matplotlib.pyplot as plt
import random

if __name__ == '__main__':
    random.seed(10)
    length = 350
    A = 50
    B = .05
    C = 5
    S = 5
    S2 = 20
    trend = [A + B * i for i in range(length)]
    seasons = [S * sin(i / 5) for i in range(length)]
    seasons2 = [S2 * sin(i / 30) for i in range(length)]
    noise = [C * random.gauss(0, 1) for _ in range(length)]
    model = [trend[i] + seasons[i] + seasons2[i] for i in range(length)]
    ts = [model[i] + noise[i] for i in range(length)]
    plt.xticks([])
    plt.yticks([])
    plt.plot(ts, '--', label = 'time series')
    plt.plot(model, linewidth = 3.0, label = 'model')
    plt.legend()
    plt.show()
