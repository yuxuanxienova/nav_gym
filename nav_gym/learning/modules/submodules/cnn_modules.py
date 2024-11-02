import torch.nn as nn
import numpy as np
import torch
from .mlp_modules import MLP

import math


class SPPooling(nn.Module):
    def __init__(self, out_pool_size):
        super().__init__()
        self.out_pool_size = out_pool_size

    def forward(self, x):
        pooled = []
        # print(x.shape)
        for i in range(len(self.out_pool_size)):
            # print(previous_conv_size)
            h_wid = int(math.ceil(x.shape[2] / self.out_pool_size[i]))
            w_wid = int(math.ceil(x.shape[3] / self.out_pool_size[i]))
            h_pad = min(int((h_wid * self.out_pool_size[i] - x.shape[2] + 1) / 2), h_wid / 2)
            w_pad = min(int((w_wid * self.out_pool_size[i] - x.shape[3] + 1) / 2), w_wid / 2)
            # print(h_wid)
            # print(h_pad)
            # exit()
            pool_op = nn.AvgPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            pooled.append(torch.flatten(pool_op(x), 1))
        return torch.cat(pooled, 1)


class SimpleCNN(nn.Module):
    def __init__(self, input_shape, channels, fc_shape, fc_output, activation_fn=nn.LeakyReLU, groups=1, kernel_size=3):
        super().__init__()
        modules = [nn.Conv2d(input_shape[0], channels[0], kernel_size, groups=groups), activation_fn()]

        for idx in range(len(channels) - 1):
            modules.append(nn.Conv2d(channels[idx], channels[idx + 1], kernel_size, groups=groups))
            modules.append(activation_fn())
            if idx > 0:
                modules.append(nn.Conv2d(channels[idx + 1], channels[idx + 1], kernel_size, kernel_size, groups=groups))
                # downsample

        # modules.append(nn.Conv2d(channels[-1], channels[-1], 2, 2))

        self.input_shape = input_shape
        self.conv_module = nn.Sequential(*modules)

        dummy_input = torch.zeros(1, input_shape[0], input_shape[1], input_shape[2])
        dummy_output = self.conv_module(dummy_input).view(1, -1)

        self.fc = MLP(fc_shape, activation_fn, dummy_output.shape[1], fc_output, 1.0 / np.sqrt(2))

    def forward(self, x):
        x = x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        x = self.conv_module(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc(x)
        return x

class SimpleCNN2(nn.Module):
    def __init__(self, input_shape, channels, fc_shape, fc_output, activation_fn=nn.LeakyReLU, groups=1, kernel_size=3, downsample_stride=2):
        super().__init__()
        modules = [nn.Conv2d(input_shape[0], channels[0], kernel_size, groups=groups), activation_fn()]

        for idx in range(len(channels) - 1):
            modules.append(nn.Conv2d(channels[idx], channels[idx + 1], kernel_size,downsample_stride, groups=groups))
            modules.append(activation_fn())

        # modules.append(nn.Conv2d(channels[-1], channels[-1], 2, 2))

        self.input_shape = input_shape
        self.conv_module = nn.Sequential(*modules)

        dummy_input = torch.zeros(1, input_shape[0], input_shape[1], input_shape[2])
        dummy_output = self.conv_module(dummy_input).view(1, -1)

        self.fc = MLP(fc_shape, activation_fn, dummy_output.shape[1], fc_output, 1.0 / np.sqrt(2))

    def forward(self, x):
        x = x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        x = self.conv_module(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc(x)
        return x


class CnnSPP(nn.Module):
    def __init__(self, input_shape, channels, pooling_size, fc_shape, fc_output, activation_fn=nn.LeakyReLU):
        super().__init__()
        modules = [nn.Conv2d(1, channels[0], 3), activation_fn()]

        for idx in range(len(channels) - 1):
            modules.append(nn.Conv2d(channels[idx], channels[idx + 1], 3))
            modules.append(activation_fn())

        self.input_shape = input_shape
        self.conv_module = nn.Sequential(*modules)
        self.pooling_size = pooling_size
        num_output_features = 0
        for level in pooling_size:
            num_output_features += channels[-1] * level * level

        self.pooling = SPPooling(pooling_size)
        self.fc = MLP(fc_shape, activation_fn, num_output_features, fc_output, 1.0 / np.sqrt(2))

    def forward(self, x):
        x = x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        x = self.conv_module(x)
        x = self.pooling.forward(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc(x)
        return x


class Pool1DConv(nn.Module):
    # input_shape = [seq_len, dim] --> has to be swapped to [dim, seq_len]
    def __init__(self, input_shape, channels, fc_shape, fc_output, kernel_size=3, activation_fn=nn.LeakyReLU):
        super().__init__()

        seq_len = input_shape[0]
        data_dim = input_shape[1]

        modules = [nn.Conv1d(data_dim, channels[0], 1), activation_fn()]

        for idx in range(len(channels) - 1):
            modules.append(nn.Conv1d(channels[idx], channels[idx + 1], kernel_size))
            modules.append(activation_fn())
        # modules.append(nn.Conv2d(channels[-1], channels[-1], 2, 2))  # downsample

        self.input_shape = input_shape
        self.conv_module = nn.Sequential(*modules)

        self.fc = MLP(fc_shape, activation_fn, channels[-1], fc_output, 1.0 / np.sqrt(2))

    def forward(self, x):
        x = x.view(-1, self.input_shape[0], self.input_shape[1])
        x = torch.swapaxes(x, 1, 2)
        x = self.conv_module(x)
        x = torch.max(x, 2, keepdim=True)[0]
        # x = torch.mean(x, 2, keepdim=True)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc(x)
        return x
