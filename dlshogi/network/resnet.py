#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn

from .entry import Entry
from .residual_block import ResidualBlock

__author__ = 'Yasuhiro'
__date__ = '2021/02/14'


class NetworkBase(nn.Module):
    def __init__(self, blocks, channels, pre_act=False, activation=nn.SiLU):
        super(NetworkBase, self).__init__()
        if pre_act:
            self.entry = Entry(out_channels=channels)
        else:
            self.entry = nn.Sequential(
                Entry(out_channels=channels),
                nn.BatchNorm2d(num_features=channels),
                activation()
            )

        self.blocks = nn.Sequential(*[
            ResidualBlock(in_channels=channels, out_channels=channels,
                          pre_act=pre_act, activation=activation)
            for _ in range(blocks)
        ])

    def forward(self, x):
        h = self.entry(x)
        y = self.blocks(h)
        return y
