#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
# noinspection PyProtectedMember
from torch.nn.modules.utils import _pair as pair

from .norm import FilterResponseNorm
from .squeeze_excitation import SqueezeExcitation

__author__ = 'Yasuhiro'
__date__ = '2021/02/14'


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pre_act=False,
                 activation=nn.SiLU):
        super(BasicBlock, self).__init__()
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


class BottleneckBlock(nn.Module):
    def __init__(self, channels, activation=nn.SiLU, expansion=4):
        super(BottleneckBlock, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(num_features=channels),
            activation(),
            nn.Conv2d(
                in_channels=channels, out_channels=channels // expansion,
                kernel_size=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(num_features=channels // expansion),
            activation(),
            nn.Conv2d(
                in_channels=channels // expansion,
                out_channels=channels // expansion,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(num_features=channels // expansion),
            activation(),
            nn.Conv2d(
                in_channels=channels // expansion, out_channels=channels,
                kernel_size=1, padding=0, bias=True
            )
        )

    def forward(self, x):
        h = self.net(x)
        y = h + x
        return y


class BottleneckBlockNext(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False,
                 rectified_conv=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, last_gamma=False):
        super(BottleneckBlockNext, self).__init__()

        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1,
                               bias=False)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock2D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock2D(dropblock_prob, 3)
            self.dropblock3 = DropBlock2D(dropblock_prob, 3)

        if radix >= 1:
            self.conv2 = SplitAttentionConv2d(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False,
                radix=radix, rectify=rectified_conv,
                rectify_avg=rectify_avg,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob
            )
        elif rectified_conv:
            from rfconv import RFConv2d
            self.conv2 = RFConv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False,
                average_mode=rectify_avg
            )
            self.bn2 = norm_layer(group_width)
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False
            )
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(
            group_width, planes * 4, kernel_size=1, bias=False
        )
        self.bn3 = norm_layer(planes * 4)

        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        h = self.conv1(x)
        h = self.bn1(h)
        if self.dropblock_prob > 0.0:
            # noinspection PyCallingNonCallable
            h = self.dropblock1(h)
        h = self.relu(h)

        if self.avd and self.avd_first:
            h = self.avd_layer(h)

        h = self.conv2(h)
        if self.radix == 0:
            h = self.bn2(h)
            if self.dropblock_prob > 0.0:
                # noinspection PyCallingNonCallable
                h = self.dropblock2(h)
            h = self.relu(h)

        if self.avd and not self.avd_first:
            h = self.avd_layer(h)

        h = self.conv3(h)
        h = self.bn3(h)
        if self.dropblock_prob > 0.0:
            # noinspection PyCallingNonCallable
            h = self.dropblock3(h)

        if self.downsample is not None:
            residual = self.downsample(x)

        h += residual
        y = self.relu(h)

        return y


class SplitAttentionConv2d(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1),
                 padding=(0, 0), dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4, rectify=False, rectify_avg=False,
                 norm_layer=None, dropblock_prob=0.0, **kwargs):
        super(SplitAttentionConv2d, self).__init__()

        padding = pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob

        if self.rectify:
            from rfconv import RFConv2d
            self.conv = RFConv2d(
                in_channels=in_channels, out_channels=channels * radix,
                kernel_size=kernel_size, stride=stride, padding=padding,
                dilation=dilation, groups=groups * radix, bias=bias,
                average_mode=rectify_avg, **kwargs
            )
        else:
            self.conv = nn.Conv2d(
                in_channels=in_channels, out_channels=channels * radix,
                kernel_size=kernel_size, stride=stride, padding=padding,
                dilation=dilation, groups=groups * radix, bias=bias,
                **kwargs
            )

        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels * radix)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(
            in_channels=channels, out_channels=inter_channels,
            kernel_size=1, groups=self.cardinality
        )
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv2d(
            in_channels=inter_channels, out_channels=channels * radix,
            kernel_size=1, groups=self.cardinality
        )
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(p=dropblock_prob)
        self.rsoftmax = rSoftMax(radix=radix, cardinality=groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            # noinspection PyCallingNonCallable
            x = self.dropblock(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            split = torch.split(x, rchannel // self.radix, dim=1)
            gap = sum(split)
        else:
            gap = x
        # noinspection PyTypeChecker
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        attention = self.fc2(gap)
        attention = self.rsoftmax(attention).view(batch, -1, 1, 1)

        if self.radix > 1:
            attentions = torch.split(attention, rchannel // self.radix, dim=1)
            # noinspection PyUnboundLocalVariable
            y = sum([a * s for a, s in zip(attentions, split)])
        else:
            y = attention * x
        return y.contiguous()


# noinspection PyPep8Naming
class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super(rSoftMax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class DropBlock2D(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class BasicBlockFRN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlockFRN, self).__init__()
        self.net = nn.Sequential(
            FilterResponseNorm(num_features=in_channels),
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=3, padding=1
            ),
            FilterResponseNorm(num_features=in_channels),
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=3, padding=1
            )
        )

    def forward(self, x):
        h = self.net(x)
        y = h + x
        return y


class BasicBlockSE(nn.Module):
    def __init__(self, channels, activation=nn.SiLU):
        super(BasicBlockSE, self).__init__()

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
        self.se = SqueezeExcitation(channels=channels, ratio=16)

    def forward(self, x):
        h = self.net(x)
        h = self.se(h)
        y = h + x
        return y


class BasicBlockFRNSE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlockFRNSE, self).__init__()

        self.net = nn.Sequential(
            FilterResponseNorm(num_features=in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, padding=1, bias=False),
            FilterResponseNorm(num_features=out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3, padding=1)
        )
        self.se = SqueezeExcitation(channels=out_channels, ratio=16)

    def forward(self, x):
        h = self.net(x)
        h = self.se(h)
        y = h + x
        return y


class BasicBlockDilationSE(nn.Module):
    def __init__(self, channels, activation=nn.SiLU):
        super(BasicBlockDilationSE, self).__init__()

        self.net1 = nn.Sequential(
            nn.BatchNorm2d(num_features=channels),
            activation(),
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=channels),
            activation(),
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=3, padding=1),
            SqueezeExcitation(channels=channels, ratio=16)
        )

        self.net2 = nn.Sequential(
            nn.BatchNorm2d(num_features=channels),
            activation(),
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(num_features=channels),
            activation(),
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=3, padding=2, dilation=2),
            SqueezeExcitation(channels=channels, ratio=16)
        )

    def forward(self, x):
        h1 = self.net1(x)
        h2 = self.net2(x)
        y = h1 + h2 + x
        return y


class InvertedBottleneckBlock(nn.Module):
    def __init__(self, channels, activation=nn.SiLU, expansion=4):
        super(InvertedBottleneckBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels * expansion,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=channels * expansion),
            activation(),
            nn.Conv2d(in_channels=channels * expansion,
                      out_channels=channels * expansion,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=channels * expansion),
            activation(),
            nn.Conv2d(in_channels=channels * expansion, out_channels=channels,
                      kernel_size=1)
        )

    def forward(self, x):
        h = self.net(x)
        y = h + x
        return y


class InvertedBottleneckBlockSD(nn.Module):
    def __init__(self, channels, activation=nn.SiLU, expansion=4,
                 n=None, total=None):
        super(InvertedBottleneckBlockSD, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels * expansion,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=channels * expansion),
            activation(),
            nn.Conv2d(in_channels=channels * expansion,
                      out_channels=channels * expansion,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=channels * expansion),
            activation(),
            nn.Conv2d(in_channels=channels * expansion, out_channels=channels,
                      kernel_size=1)
        )

        self.probability = 1 - n / (total - 1) * 0.5

    def forward(self, x):
        if self.training:
            if np.random.rand() < self.probability:
                self.net.requires_grad_(True)
                h = self.net(x)
                y = h + x
            else:
                self.net.requires_grad_(False)
                y = x
        else:
            h = self.net(x) * self.probability
            y = h + x

        return y
