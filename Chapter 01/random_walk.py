import matplotlib.pyplot as plt
import random


def generate_random_walk(length = 100, mu = 0, sig = 1):
    ts = []
    for i in range(length):
        e = random.gauss(mu, sig)
        if i == 0:
            ts.append(e)
        else:
            ts.append(ts[i - 1] + e)
    return ts


if __name__ == '__main__':
    random.seed(10)
    random_walk = generate_random_walk(100)
    plt.plot(random_walk)
    plt.show()
