#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn

from .entry import Entry
from .residual_block import (
    BasicBlock, BottleneckBlockNext, BasicBlockFRN, BasicBlockSE,
    BasicBlockDilationSE, BottleneckBlock, InvertedBottleneckBlock,
    InvertedBottleneckBlockSD
)

__author__ = 'Yasuhiro'
__date__ = '2021/02/14'


class NetworkBase(nn.Module):
    def __init__(self, blocks, channels, pre_act=False, activation=nn.SiLU,
                 squeeze_excitation=True, bottleneck=True,
                 bottleneck_expansion=4, **kwargs):
        super(NetworkBase, self).__init__()
        if pre_act:
            self.entry = Entry(out_channels=channels)
        else:
            self.entry = nn.Sequential(
                Entry(out_channels=channels),
                nn.BatchNorm2d(num_features=channels),
                activation()
            )

        if squeeze_excitation:
            assert pre_act, "pre activation only!"
            self.blocks = nn.Sequential(*[
                BasicBlockSE(channels=channels, activation=activation)
                for _ in range(blocks)
            ])
        else:
            if bottleneck:
                self.blocks = nn.Sequential(*[
                    InvertedBottleneckBlock(
                        channels=channels, activation=activation,
                        expansion=bottleneck_expansion
                    )
                    for _ in range(blocks)
                ])
            else:
                self.blocks = nn.Sequential(*[
                    BasicBlock(in_channels=channels, out_channels=channels,
                               pre_act=pre_act, activation=activation)
                    for _ in range(blocks)
                ])

    def forward(self, x):
        h = self.entry(x)
        y = self.blocks(h)
        return y


class NetworkBaseFRN(nn.Module):
    def __init__(self, blocks, channels, **kwargs):
        super(NetworkBaseFRN, self).__init__()
        self.entry = Entry(out_channels=channels)

        self.blocks = nn.Sequential(*[
            BasicBlockFRN(in_channels=channels, out_channels=channels)
            for _ in range(blocks)
        ])

    def forward(self, x):
        h = self.entry(x)
        y = self.blocks(h)
        return y


class NetworkBaseNext(nn.Module):
    def __init__(self, blocks, channels, radix=1, groups=1,
                 bottleneck_width=64,  rectified_conv=False, rectify_avg=False,
                 norm_layer=nn.BatchNorm2d, activation=nn.SiLU, **kwargs):
        super(NetworkBaseNext, self).__init__()

        layers = [
            Entry(out_channels=channels),
            norm_layer(channels),
            activation()
        ]
        layers.extend([
            BottleneckBlockNext(
                inplanes=channels, planes=channels // 4,
                radix=radix, cardinality=groups,
                bottleneck_width=bottleneck_width,
                rectified_conv=rectified_conv, rectify_avg=rectify_avg,
                norm_layer=norm_layer
            ) for _ in range(blocks)
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        return y


class NetworkBaseStochasticDepth(nn.Module):
    def __init__(self, blocks, channels, activation=nn.SiLU,
                 bottleneck=True, bottleneck_expansion=4, **kwargs):
        super(NetworkBaseStochasticDepth, self).__init__()
        assert bottleneck

        self.entry = Entry(out_channels=channels)
        self.blocks = nn.Sequential(*[
            InvertedBottleneckBlockSD(
                channels=channels, activation=activation,
                expansion=bottleneck_expansion, n=i, total=blocks
            )
            for i in range(blocks)
        ])

    def forward(self, x):
        h = self.entry(x)
        y = self.blocks(h)
        return y
