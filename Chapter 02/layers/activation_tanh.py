import torch
import matplotlib.pyplot as plt

x = torch.linspace(-10, 10)
relu = torch.nn.Tanh()
y = relu(x)

plt.title('Tanh')
plt.plot(x.tolist(), y.tolist())
plt.show()
