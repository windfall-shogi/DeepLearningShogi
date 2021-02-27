#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn

__author__ = 'Yasuhiro'
__date__ = '2021/02/27'


class XceptionMiddle(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.SiLU):
        super(XceptionMiddle, self).__init__()
        self.activation = activation

        self.net = nn.Sequential(
            activation(),
            SeparableConv2d(in_channels=in_channels,
                            out_channels=out_channels),
            activation(),
            SeparableConv2d(in_channels=in_channels,
                            out_channels=out_channels),
            activation(),
            SeparableConv2d(in_channels=in_channels,
                            out_channels=out_channels)
        )

        if in_channels != out_channels:
            self.bypass = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=1
            )
        else:
            self.bypass = nn.Identity()

    def forward(self, x):
        h = self.net(x)
        g = self.bypass(x)
        y = h + g
        return y


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SeparableConv2d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 4,
                      kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels=in_channels * 4, out_channels=out_channels,
                      kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)
