import random
import numpy as np


def generate_time_series(len):
    backshift = 10
    r1 = np.random.random(len + backshift)
    r2 = np.random.random(len + backshift)
    rm = [random.choices([0, 0, 0, 1])[0]
          for _ in range(len + backshift)]

    ts = np.zeros([len + backshift, 4])
    for i in range(backshift, len + backshift):
        ts[i - 1, 1] = r1[i - 1]
        ts[i - 1, 2] = r2[i - 1]
        ts[i - 1, 3] = rm[i - 1]

        ts[i, 0] = ts[i - 1, 0] -\
                   (r1[i - 1] + r1[i - 2]) +\
                   4 * r2[i - 3] * (rm[i - 4] + rm[i - 6])

    return ts[backshift:]
