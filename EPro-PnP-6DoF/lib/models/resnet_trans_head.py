"""
This file is from
https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi
"""

import torch.nn as nn
import torch


class TransHeadNet(nn.Module):
    def __init__(self, in_channels, num_layers=3, num_filters=256, kernel_size=3, output_dim=3, freeze=False,
                 with_bias_end=True):
        super(TransHeadNet, self).__init__()

        self.freeze = freeze

        if kernel_size == 3:
            padding = 1
        elif kernel_size == 2:
            padding = 0

        self.features = nn.ModuleList()
        for i in range(num_layers):
            _in_channels = in_channels if i == 0 else num_filters
            self.features.append(nn.Conv2d(_in_channels, num_filters, kernel_size=kernel_size, stride=1, padding=padding, bias=False))
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.ReLU(inplace=True))

        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(256 * 8 * 8, 4096))
        self.linears.append(nn.ReLU(inplace=True))
        self.linears.append(nn.Linear(4096, 4096))
        self.linears.append(nn.ReLU(inplace=True))
        self.linears.append(nn.Linear(4096, output_dim))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if with_bias_end and (m.bias is not None):
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                for i, l in enumerate(self.features):
                    x = l(x)
                x = x.view(-1, 256*8*8)
                for i, l in enumerate(self.linears):
                    x = l(x)
                return x.detach()
        else:
            for i, l in enumerate(self.features):
                x = l(x)
            x = x.view(-1, 256*8*8)
            for i, l in enumerate(self.linears):
                x = l(x)
            return x
