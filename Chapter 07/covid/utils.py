from datetime import datetime
import torch


def sliding_window(ts, features, target_len = 1):
    X, Y = [], []
    for i in range(features + target_len, len(ts) + 1):
        X.append(ts[i - (features + target_len):i - target_len])
        Y.append(ts[i - target_len:i])
    return X, Y


def get_date_index(ts, date_index):
    count = 0
    for i, r in ts.iteritems():
        if date_index == datetime.strftime(i, '%Y-%m-%d'):
            return count
        count = count + 1
    return -1


def get_train_test_datasets(ts, w, target_len, train_dates, test_dates):
    X, Y = sliding_window(ts, w, target_len)
    train_from = get_date_index(ts, train_dates[0]) - w
    train_to = get_date_index(ts, train_dates[1]) - w
    test_from = get_date_index(ts, test_dates[0]) - w
    test_to = get_date_index(ts, test_dates[1]) - w

    x_train = torch.tensor(data = X[train_from:train_to]).float()
    y_train = torch.tensor(data = Y[train_from:train_to]).float()

    x_test = torch.tensor(data = X[test_from:test_to]).float()
    y_test = torch.tensor(data = Y[test_from:test_to]).float()

    return x_train, x_test, y_train, y_test
