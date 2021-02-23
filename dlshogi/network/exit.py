#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn

from ..common import MAX_MOVE_LABEL_NUM

__author__ = 'Yasuhiro'
__date__ = '2021/02/21'


class Policy(nn.Module):
    def __init__(self, in_channels):
        super(Policy, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=MAX_MOVE_LABEL_NUM,
                kernel_size=1, bias=False
            ),
            Reshape(-1, MAX_MOVE_LABEL_NUM * 9 * 9),
            Bias(MAX_MOVE_LABEL_NUM * 9 * 9)
        )

    def forward(self, x):
        return self.net(x)


class Value(nn.Module):
    def __init__(self, in_channels, num_features, activation):
        super(Value, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.Conv2d(
                in_channels=in_channels, out_channels=MAX_MOVE_LABEL_NUM,
                kernel_size=1, bias=False
            ),
            activation(),
            Reshape(-1, MAX_MOVE_LABEL_NUM * 9 * 9),
            nn.Linear(in_features=MAX_MOVE_LABEL_NUM * 9 * 9,
                      out_features=num_features),
            activation(),
            nn.Linear(in_features=num_features, out_features=1)
        )

    def forward(self, x):
        return self.net(x)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Bias(nn.Module):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.Tensor(shape))

    def forward(self, x):
        return x + self.bias
