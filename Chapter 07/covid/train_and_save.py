from ch7.covid.train import prepare_model

import random
import numpy as np
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# NNI result:
# Best trial params: {'hidden_size': 32, 'hidden_dl_size': 12, 'lr': 0.01, 'tfr': 0.1}
params = {
    'hidden_size':    32,
    'hidden_dl_size': 12,
    'lr':             .01,
    'tfr':            .1,
}

prepare_model(params, save_model = True, model_name = 'best_model')
