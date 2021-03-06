#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn

__author__ = 'Yasuhiro'
__date__ = '2021/03/06'


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, inputs):
        # noinspection PyTypeChecker
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(
            inputs.size(0), -1)
