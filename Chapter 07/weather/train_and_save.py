from ch7.weather.train import prepare_model
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
    "tcl_num":          2,
    "tcl_channel_size": 32,
    "kernel_size":      7,
    "dropout":          0.1,
    "slices":           1,
    "use_bias":         True,
    "lr":               0.005
}

prepare_model(params, save_model = True, model_name = 'best_model')
