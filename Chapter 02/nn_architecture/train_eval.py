from ch2.nn_architecture.mnist import MnistModel

net = MnistModel()

print(f'Training mode enabled: {net.training}')
net.eval()
print(f'Training mode enabled: {net.training}')
net.train()
print(f'Training mode enabled: {net.training}')
