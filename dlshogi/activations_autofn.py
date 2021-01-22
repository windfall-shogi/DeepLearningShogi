import torch
from torch import nn as nn
from torch.nn import functional as F

"""
original code: https://github.com/digantamisra98/Mish/blob/master/PyTorch%20Benchmarks/activations_autofn.py
license: MIT : https://github.com/digantamisra98/Mish/blob/master/LICENSE
"""


__all__ = ['swish_auto', 'SwishAuto', 'Swish', 'mish_auto', 'MishAuto', 'Mish', 'tanhexp_auto', 'TanhExpAuto', 'TanhExp']


# in-place Swish oper
class SwishAutoFn(torch.autograd.Function):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    Memory efficient variant from:
     https://medium.com/the-artificial-impostor/more-memory-efficient-swish-activation-function-e07c22c12a76
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sigmoid().mul_(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        x_sigmoid = x.sigmoid()
        return x_sigmoid.neg().add_(1).mul_(x).add_(1).mul_(x_sigmoid).mul_(grad_output)


# normal Swish module (for ONNX export)
class Swish(nn.Module):
    def forward(self, x):
        return x.sigmoid().mul_(x)


# in-place Swish function
def swish_auto(x, inplace=False):
    # inplace ignored
    return SwishAutoFn.apply(x)


# in-place Swish module
class SwishAuto(nn.Module):
    def __init__(self, inplace: bool = True):
        super(SwishAuto, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return SwishAutoFn.apply(x)


# in-place Mish oper
class MishAutoFn(torch.autograd.Function):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    Experimental memory-efficient variant
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return F.softplus(x).tanh_().mul_(x)  # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        x_tanh_sp = F.softplus(x).tanh_()
        return x_tanh_sp.square().neg_().add_(1).mul_(x.sigmoid()).mul_(x).add_(x_tanh_sp).mul_(grad_output)


# normal Mish module (for ONNX export)
class Mish(nn.Module):
    def forward(self, x):
        return F.softplus(x).tanh_().mul_(x)  # x * tanh(ln(1 + exp(x)))


# in-place Mish function
def mish_auto(x, inplace=False):
    # inplace ignored
    return MishAutoFn.apply(x)


# in-place Mish module
class MishAuto(nn.Module):
    def __init__(self, inplace: bool = True):
        super(MishAuto, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return MishAutoFn.apply(x)


# in-place TanhExp oper
class TanhExpAutoFn(torch.autograd.Function):
    """TanhExp - https://arxiv.org/abs/2003.09855
    A Smooth Activation Function with High Convergence Speed for Lightweight Neural Networks
    Experimental memory-efficient variant
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(max=6).exp_().tanh_().mul_(x)  # x * tanh(exp(x))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        x_exp = x.clamp(max=6).exp_()
        x_exp_tanh = x_exp.tanh()
        return x_exp_tanh.square().neg_().add_(1).mul_(x_exp).mul_(x).add_(x_exp_tanh).mul_(grad_output)


# normal TanhExp module (for ONNX export)
class TanhExp(nn.Module):
    def forward(self, x):
        return x.clamp(max=6).exp_().tanh_().mul_(x)  # x * tanh(exp(x))


# in-place TanhExp function
def tanhexp_auto(x, inplace=False):
    # inplace ignored
    return TanhExpAutoFn.apply(x)


# in-place TanhExp module
class TanhExpAuto(nn.Module):
    def __init__(self, inplace: bool = True):
        super(TanhExpAuto, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return TanhExpAutoFn.apply(x)
