def sliding_window(ts, features):
    X = []
    Y = []

    for i in range(features + 1, len(ts) + 1):
        X.append(ts[i - (features + 1):i - 1])
        Y.append([ts[i - 1]])

    return X, Y


if __name__ == '__main__':
    ts = list(range(6))
    X, Y = sliding_window(ts, 3)

    print(f'Time series: {ts}')
    print(f'X: {X}')
    print(f'Y: {Y}')
