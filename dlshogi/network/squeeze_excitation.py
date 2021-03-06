#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn

from .pooling import GlobalAvgPool2d

__author__ = 'Yasuhiro'
__date__ = '2021/03/06'


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, ratio=8):
        super(SqueezeExcitation, self).__init__()
        self.spatial = SpatialSE(channels=channels, ratio=ratio)
        self.depth = DepthSE(channels=channels)

    def forward(self, x):
        return self.spatial(x) + self.depth(x)


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
        batch = h.size(0)
        h = h.view(batch, -1, 1, 1)

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
