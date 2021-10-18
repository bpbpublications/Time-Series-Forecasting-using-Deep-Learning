import torch
import matplotlib.pyplot as plt

x = torch.linspace(-10, 10)
relu = torch.nn.ReLU()
y = relu(x)

plt.title('ReLU')
plt.plot(x.tolist(), y.tolist())
plt.show()
