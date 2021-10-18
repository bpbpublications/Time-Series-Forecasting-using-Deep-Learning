import torch
import matplotlib.pyplot as plt

x = torch.linspace(-10, 10)
relu = torch.nn.Sigmoid()
y = relu(x)

plt.title('Sigmoid')
plt.plot(x.tolist(), y.tolist())
plt.show()
