def sliding_window(ts, features, target_len = 1):
    X, Y = [], []
    for i in range(features + target_len, len(ts) + 1):
        X.append(ts[i - (features + target_len):i - target_len])
        Y.append(ts[i - target_len:i])
    return X, Y
