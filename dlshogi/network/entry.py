#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn

from ..common import FEATURES1_NUM, FEATURES2_NUM

__author__ = 'Yasuhiro'
__date__ = '2021/02/14'


class Entry(nn.Module):
    def __init__(self, out_channels, bias=False):
        super(Entry, self).__init__()
        self.in1a = nn.Conv2d(
            in_channels=FEATURES1_NUM, out_channels=out_channels,
            kernel_size=3, padding=1, bias=bias
        )
        self.in1b = nn.Conv2d(
            in_channels=FEATURES1_NUM, out_channels=out_channels,
            kernel_size=1, padding=0, bias=bias
        )
        self.in2 = nn.Conv2d(
            in_channels=FEATURES2_NUM, out_channels=out_channels,
            kernel_size=1, padding=0, bias=bias
        )

    def forward(self, x):
        x1, x2 = x
        h1a = self.in1a(x1)
        h1b = self.in1b(x1)
        h2 = self.in2(x2)
        y = h1a + h1b + h2
        return y
