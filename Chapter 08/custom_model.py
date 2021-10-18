import torch
import torch.nn.functional as F


class CustomModel(torch.nn.Module):

    def __init__(self, n_inp, l_1, l_2, conv1_out, conv1_kernel, conv2_kernel, drop1 = 0):
        super(CustomModel, self).__init__()
        conv1_out_ch = conv1_out
        conv2_out_ch = conv1_out * 2
        conv1_kernel = conv1_kernel
        conv2_kernel = conv2_kernel
        self.dropout_lin1 = drop1

        self.pool = torch.nn.MaxPool1d(kernel_size = 2)

        self.conv1 = torch.nn.Conv1d(in_channels = 1, out_channels = conv1_out_ch, kernel_size = conv1_kernel,
                                     padding = conv1_kernel - 1)

        self.conv2 = torch.nn.Conv1d(in_channels = conv1_out_ch, out_channels = conv2_out_ch,
                                     kernel_size = conv2_kernel,
                                     padding = conv2_kernel - 1)

        feature_tensor = self.feature_stack(torch.Tensor([[0] * n_inp]))

        self.lin1 = torch.nn.Linear(feature_tensor.size()[1], l_1)
        self.lin2 = torch.nn.Linear(l_1, l_2)
        self.lin3 = torch.nn.Linear(l_2, 1)

    def feature_stack(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = x.flatten(start_dim = 1)
        return x

    def fc_stack(self, x):
        x1 = F.dropout(F.relu(self.lin1(x)), p = self.dropout_lin1)
        x2 = F.relu(self.lin2(x1))
        y = self.lin3(x2)
        return y

    def forward(self, x):
        x = self.feature_stack(x)
        y = self.fc_stack(x)
        return y
