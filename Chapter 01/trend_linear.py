import matplotlib.pyplot as plt
import random

if __name__ == '__main__':
    random.seed(10)
    length = 50
    A = 5
    B = .5
    C = 3
    trend = [A + B * i for i in range(length)]
    noise = [C * random.gauss(0, 1) for _ in range(length)]
    ts = [trend[i] + noise[i] for i in range(length)]
    plt.plot(ts)
    plt.plot(trend)
    plt.show()
