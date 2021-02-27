#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn

from .entry import Entry
from .xception_block import XceptionMiddle

__author__ = 'Yasuhiro'
__date__ = '2021/02/27'


class NetworkBase(nn.Module):
    def __init__(self, blocks, channels, activation=nn.SiLU, **kwargs):
        super(NetworkBase, self).__init__()

        self.entry = Entry(out_channels=channels)
        self.blocks = nn.Sequential(*[
            XceptionMiddle(in_channels=channels, out_channels=channels,
                           activation=activation)
            for _ in range(blocks)
        ])

    def forward(self, x):
        h = self.entry(x)
        y = self.blocks(h)
        return y
