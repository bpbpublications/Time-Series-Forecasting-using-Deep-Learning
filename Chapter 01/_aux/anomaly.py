import matplotlib.pyplot as plt
import random

if __name__ == '__main__':

    random.seed(9)
    length = 100
    A = 5
    B = .2
    C = 1
    trend = [A + B * i for i in range(length)]
    noise = []
    for i in range(length):
        if 65 <= i <= 75:
            noise.append(7 * C * random.gauss(0, 1))
            plt.axvspan(i, i + 1, color = 'red', alpha = 0.1)
        else:
            noise.append(C * random.gauss(0, 1))

    ts = [trend[i] + noise[i] for i in range(length)]
    plt.plot(ts)
    plt.xticks([])
    plt.yticks([])
    plt.show()
