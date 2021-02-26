#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn

__author__ = 'Yasuhiro'
__date__ = '2021/02/14'


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pre_act=False,
                 activation=nn.SiLU):
        super(ResidualBlock, self).__init__()
        self.pre_act = pre_act
        self.activation = activation
        if pre_act:
            self.net = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                activation(),
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(num_features=out_channels),
                activation(),
                nn.Conv2d(
                    in_channels=out_channels, out_channels=out_channels,
                    kernel_size=3, padding=1, bias=True
                )
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(num_features=out_channels),
                activation(),
                nn.Conv2d(
                    in_channels=out_channels, out_channels=out_channels,
                    kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(num_features=out_channels)
            )
            self.post_activator = self.activation()

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
        if self.pre_act:
            y = h + g
        else:
            y = self.post_activator(h + g)
        return y
