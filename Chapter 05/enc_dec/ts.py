import numpy as np
import matplotlib.pyplot as plt


def generate_ts(len):
    tf = 80 * np.pi
    t = np.linspace(0., tf, len)
    y = np.sin(t) + 0.8 * np.cos(.5 * t) + np.random.normal(0., 0.3, len) + 2.5
    return y.tolist()


if __name__ == '__main__':
    ts = generate_ts(2000)
    plt.plot(ts[:300])
    plt.show()
