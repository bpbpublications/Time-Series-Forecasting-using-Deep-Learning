from math import log

import matplotlib.pyplot as plt
import random

if __name__ == '__main__':
    random.seed(1)
    length = 100
    A = 2
    B = 25
    C = 5
    noise = [C * random.gauss(0, 1) for _ in range(length)]
    trend = [A + B * log(i) for i in range(1, length + 1)]
    ts = [trend[i] + noise[i] for i in range(length)]
    plt.plot(ts)
    plt.plot(trend)
    plt.show()
