#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn

from ..common import MAX_MOVE_LABEL_NUM
from .residual_block import GlobalAvgPool2d

__author__ = 'Yasuhiro'
__date__ = '2021/02/21'


class Policy(nn.Module):
    def __init__(self, in_channels):
        super(Policy, self).__init__()
        self.net = nn.Sequential(
            ResidualBlockSE(channels=in_channels),
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


class ResidualBlockSE(nn.Module):
    def __init__(self, channels, activation=nn.SiLU):
        super(ResidualBlockSE, self).__init__()

        self.net = nn.Sequential(
            nn.BatchNorm2d(num_features=channels),
            activation(),
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=channels),
            activation(),
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=3, padding=1)
        )
        self.s_se = SpatialSE(channels=channels, ratio=16)
        self.d_se = DepthSE(channels=channels)

    def forward(self, x):
        h = self.net(x)
        h_s = self.s_se(h)
        h_d = self.d_se(h)
        y = (h_s + h_d) + x
        return y


class SpatialSE(nn.Module):
    def __init__(self, channels, ratio=8):
        super(SpatialSE, self).__init__()
        self.net = nn.Sequential(
            GlobalAvgPool2d(),
            nn.Linear(in_features=channels, out_features=channels // ratio),
            nn.ReLU(),
            nn.Linear(in_features=channels // ratio, out_features=channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.net(x)
        y = h * x
        return y


class DepthSE(nn.Module):
    def __init__(self, channels):
        super(DepthSE, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.net(x)
        y = h * x
        return y
