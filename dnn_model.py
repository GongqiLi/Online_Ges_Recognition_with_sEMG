import torch
import torch.nn as nn
import numpy as np


class DeepNeuralNet(nn.Module):
    def __init__(self, num_of_ges, batch_size):
        super(DeepNeuralNet, self).__init__()
        self.C = 4
        self.batch_size = batch_size
        self.linear_param = 640

        self._conv1 = nn.Conv2d(1, self.C, kernel_size=(3, 5), padding=(1, 2))
        self._batch_norm1 = nn.BatchNorm2d(self.C)
        self._prelu1 = nn.PReLU(self.C)
        self._dropout1 = nn.Dropout2d(0.5)
        self._pool1 = nn.MaxPool2d(kernel_size=(1, 2))

        self._conv2 = nn.Conv2d(self.C, 2 * self.C, kernel_size=(3, 5), padding=(1, 2))
        self._batch_norm2 = nn.BatchNorm2d(2 * self.C)
        self._prelu2 = nn.PReLU(2 * self.C)
        self._dropout2 = nn.Dropout2d(0.5)
        self._pool2 = nn.MaxPool2d(kernel_size=(1, 5))

        self._fc1 = nn.Linear(self.linear_param, 16)
        self._batch_norm3 = nn.BatchNorm1d(16)
        self._prelu3 = nn.PReLU(16)
        self._dropout3 = nn.Dropout(0.5)
        self._output = nn.Linear(16, num_of_ges)

        self.initialize_weights()

        print(self)
        print("Number Parameters: ", self.get_n_params())

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def init_weights(self):
        for m in self.modules():
            torch.nn.init.kaiming_normal(m.weight)
            m.bias.data.zero_()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        # print(list(x.size()))
        x = self._conv1(x)
        # print(list(x.size()))
        x = self._batch_norm1(x)
        # print(list(x.size()))
        x = self._prelu1(x)
        # print(list(x.size()))
        x = self._dropout1(x)
        # print(list(x.size()))
        x = self._pool1(x)
        # print(list(x.size()))
        x = self._conv2(x)
        # print(list(x.size()))
        x = self._batch_norm2(x)
        # print(list(x.size()))
        x = self._prelu2(x)
        # print(list(x.size()))
        x = self._dropout2(x)
        # print(list(x.size()))
        x = self._pool2(x)
        # print(list(x.size()))
        x = x.view(-1, self.linear_param)
        # print(list(x.size()))
        x = self._fc1(x)
        # print(list(x.size()))
        x = self._batch_norm3(x)
        # print(list(x.size()))
        x = self._prelu3(x)
        # print(list(x.size()))
        x = self._dropout3(x)
        # print(list(x.size()))
        x = self._output(x)
        # print(list(x.size()))
        return x

