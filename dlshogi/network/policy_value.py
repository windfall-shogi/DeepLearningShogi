#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn

from .resnet import NetworkBase
# from .resnet import NetworkBaseStochasticDepth as NetworkBase
# from .resnet import NetworkBaseFRN as NetworkBase
# from .resnet import NetworkBaseNext as NetworkBase
# from .xception import NetworkBase
from .exit import Policy, Value

__author__ = 'Yasuhiro'
__date__ = '2021/02/21'


class PolicyValueNetwork(nn.Module):
    def __init__(self, blocks, channels, features, pre_act=False,
                 activation=nn.SiLU, squeeze_excitation=True, bottleneck=True,
                 bottleneck_expansion=4, **kwargs):
        super(PolicyValueNetwork, self).__init__()
        self.base = NetworkBase(
            blocks=blocks, channels=channels, pre_act=pre_act,
            activation=activation, squeeze_excitation=squeeze_excitation,
            bottleneck=bottleneck, bottleneck_expansion=bottleneck_expansion,
            **kwargs
        )
        self.policy_network = Policy(in_channels=channels, **kwargs)
        self.value_network = Value(
            in_channels=channels, num_features=features, activation=activation
        )

    def forward(self, x):
        h = self.base(x)
        policy = self.policy_network(h)
        value = self.value_network(h)
        return policy, value
