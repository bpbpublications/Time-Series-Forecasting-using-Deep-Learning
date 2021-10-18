from ch2.nn_architecture.mnist import MnistModel

net = MnistModel()

for name, layer in net.named_children():
    print(f'{name}: {layer}')
