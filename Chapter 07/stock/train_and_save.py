from ch7.stock.train import prepare_model

import random
import numpy as np
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# NNI result:
# Best trial params:
params = {
    "lr":              0.01,
    "rnn_type":        "rnn",
    "rnn_hidden_size": 24,
    "ind_hidden_size": 1,
    "des_size":        2,
    "ind1":            {
        "_name":  "rsi",
        "length": 20
    },
    "ind2":            {
        "_name":  "cmo",
        "length": 20
    }
}

prepare_model(params, save_model = True, model_name = 'best_model')
