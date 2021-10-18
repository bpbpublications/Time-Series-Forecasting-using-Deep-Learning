import torch


def get_function(x1_val = 0, x2_val = 0, x3_val = 0, x4_val = 0):
    # variables
    x1 = torch.tensor(x1_val, requires_grad = True, dtype = torch.float32)
    x2 = torch.tensor(x2_val, requires_grad = True, dtype = torch.float32)
    x3 = torch.tensor(x3_val, requires_grad = True, dtype = torch.float32)
    x4 = torch.tensor(x4_val, requires_grad = True, dtype = torch.float32)

    # function
    p1 = x1.pow(3)
    m1 = p1 * x2
    m2 = x3 * x4
    f = m1 + m2

    vars = {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4}

    return f, vars


if __name__ == '__main__':
    f, _ = get_function(2, 4, 3, 5)
    print(f.item())
