import random
import torch
from pytorch_forecasting import DeepAR
from ch8.dataset import get_weather_dataset

random.seed(1)
torch.manual_seed(1)

dataset = get_weather_dataset()
model = DeepAR.from_dataset(dataset = dataset)

print('DeepAR model summary:')
print(model.summarize("full"))
print('==============')

print('DeepAR model hyper-parameters:')
print(model.hparams)
print('==============')

print('DeepAR model forward method execution:')
dataloader = dataset.to_dataloader(batch_size = 8)
x, y = next(iter(dataloader))
p = model(x)
print(p)
print('==============')
