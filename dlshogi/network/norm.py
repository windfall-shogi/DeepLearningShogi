#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import Parameter

__author__ = 'Yasuhiro'
__date__ = '2021/03/03'


class FilterResponseNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(FilterResponseNorm, self).__init__()
        self.num_features = num_features
        self.tau = Parameter(torch.Tensor(num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(num_features, 1, 1))
        self.eps = Parameter(torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.tau)
        nn.init.zeros_(self.beta)
        nn.init.ones_(self.gamma)

    def forward(self, inputs):
        nu2 = torch.mean(inputs ** 2, dim=(2, 3), keepdim=True, out=None)
        inputs = inputs * torch.rsqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.gamma * inputs + self.beta, self.tau)

    def extra_repr(self):
        return '{}'.format(self.num_features)
