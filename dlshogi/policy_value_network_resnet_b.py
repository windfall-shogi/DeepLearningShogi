import torch
import torch.nn as nn
import torch.nn.functional as F
import re

from dlshogi.common import *
from activations_autofn import MishAuto, Mish, SwishAuto, Swish, TanhExpAuto, TanhExp

class PolicyBias(nn.Module):
    def __init__(self, shape):
        super(PolicyBias, self).__init__()
        self.bias=nn.Parameter(torch.Tensor(shape))

    def forward(self, input):
        return input + self.bias

class ResBlockPlain(nn.Module):
    def __init__(self, in_channels, out_channels, act=nn.ReLU(inplace=True), bias=False):
        super(ResBlockPlain, self).__init__()
        channels = out_channels
        # 3x3
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=3, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(channels, eps=2e-05)
        # 3x3
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=out_channels, kernel_size=3, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=2e-05)
        # shortcut
        self.shortcut = self._shortcut(in_channels, out_channels)
        # act
        self.act = act
    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.bn2(h)
        shortcut = self.shortcut(x)
        y = self.act(h + shortcut) # skip connection
        return y
    def _shortcut(self, in_channels, out_channels):
        if in_channels != out_channels:
            return self._projection(in_channels, out_channels)
        else:
            return lambda x: x
    def _projection(self, in_channels, out_channels):
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)

class ResBlockBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, act=nn.ReLU(inplace=True), bias=False):
        super(ResBlockBottleneck, self).__init__()
        channels = out_channels
        # 1x1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(channels, eps=2e-05)
        # 3x3
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(channels, eps=2e-05)
        # 1x1
        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=out_channels, kernel_size=1, padding=0, bias=bias)
        self.bn3 = nn.BatchNorm2d(out_channels, eps=2e-05)
        # shortcut
        self.shortcut = self._shortcut(in_channels, out_channels)
        # act
        self.act = act
    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.act(h)
        h = self.conv3(h)
        h = self.bn3(h)
        shortcut = self.shortcut(x)
        y = self.act(h + shortcut) # skip connection
        return y
    def _shortcut(self, in_channels, out_channels):
        if in_channels != out_channels:
            return self._projection(in_channels, out_channels)
        else:
            return lambda x: x
    def _projection(self, in_channels, out_channels):
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)

class ResNet(nn.Module):
    def __init__(self,
                 act=nn.ReLU(inplace=True),
                 b1_layer=10,
                 b1_channels=192,
                 fcl_channels=256):
        super(ResNet, self).__init__()
        self.act = act
        self.lf_1_1 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=b1_channels, kernel_size=3, padding=1, bias=False)
        self.lf_1_2 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=b1_channels, kernel_size=1, padding=0, bias=False)
        self.lf_2 = nn.Conv2d(in_channels=FEATURES2_NUM, out_channels=b1_channels, kernel_size=1, bias=False) # pieces_in_hand
        self.norm_f = nn.BatchNorm2d(b1_channels)
        # Block 1
        self.block1 = nn.ModuleList([
            self._building_block(b1_channels) for _ in range(b1_layer)
        ])
        # policy network
        self.lp_1 = nn.Conv2d(in_channels=b1_channels, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.lp_2 = PolicyBias(9*9*MAX_MOVE_LABEL_NUM)
        # value network
        self.lv_1 = nn.Conv2d(in_channels=b1_channels, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.lv_2 = nn.Linear(9*9*MAX_MOVE_LABEL_NUM, fcl_channels)
        self.lv_3 = nn.Linear(fcl_channels, 1)
        self.norm_v = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM)

    def __call__(self, x1, x2):
        uf_1_1 = self.lf_1_1(x1)
        uf_1_2 = self.lf_1_2(x1)
        uf_2 = self.lf_2(x2)
        h = self.act(self.norm_f(uf_1_1 + uf_1_2 + uf_2))
        # block 1
        for block in self.block1:
            h = block(h)
        # policy network
        hp_1 = self.lp_1(h)
        hp_2 = self.lp_2(hp_1.view(-1, 9*9*MAX_MOVE_LABEL_NUM))
        # value network
        hv_1 = self.act(self.norm_v(self.lv_1(h)))
        hv_2 = self.act(self.lv_2(hv_1.view(-1, 9*9*MAX_MOVE_LABEL_NUM)))
        return hp_2, self.lv_3(hv_2)

    def _building_block(self,
                        in_channels,
                        out_channels=None,
                        bias=False):
        if out_channels is None:
            out_channels = in_channels
        return ResBlockPlain(in_channels, out_channels, act=self.act, bias=bias)

# ONNX出力用、value networkにsigmoid演算を付加
class ResNetAddSigmoid(ResNet):
    def __init__(self,
                 act=nn.ReLU(inplace=True),
                 b1_layer=10,
                 b1_channels=192,
                 fcl_channels=256):
        super(ResNetAddSigmoid, self).__init__(
            act=act,
            b1_layer=b1_layer,
            b1_channels=b1_channels,
            fcl_channels=fcl_channels)

    def __call__(self, x1, x2):
        y1, y2 = super(ResNetAddSigmoid, self).__call__(x1, x2)
        return y1, torch.sigmoid(y2)

# 学習用オブジェクト生成
# 省メモリ化のため、活性化関数にin-place演算を用いる
def getPolicyValueNetwork(network):
    match = re.fullmatch(r'resnet(\d+)ch(\d+)_(relu|mish|swish|tanhexp)', network)
    if match is None:
        return None
    elif match.group(3) == 'relu':
        return ResNet(
            b1_layer=int(match.group(1)),
            b1_channels=int(match.group(2)),
            act=nn.ReLU(inplace=True),
            fcl_channels=256)
    elif match.group(3) == 'mish':
        return ResNet(
            b1_layer=int(match.group(1)),
            b1_channels=int(match.group(2)),
            act=MishAuto(),
            fcl_channels=256)
    elif match.group(3) == 'swish':
        return ResNet(
            b1_layer=int(match.group(1)),
            b1_channels=int(match.group(2)),
            act=SwishAuto(),
            fcl_channels=256)
    elif match.group(3) == 'tanhexp':
        return ResNet(
            b1_layer=int(match.group(1)),
            b1_channels=int(match.group(2)),
            act=TanhExpAuto(),
            fcl_channels=256)
    else:
        return None

# ONNX出力用オブジェクト生成
# ONNXエクスポート用に、value networkの出力にsigmoid演算を付加する
def getPolicyValueNetworkAddSigmoid(network):
    match = re.fullmatch(r'resnet(\d+)ch(\d+)_(relu|mish|swish|tanhexp)', network)
    if match is None:
        return None
    elif match.group(3) == 'relu':
        return ResNetAddSigmoid(
            b1_layer=int(match.group(1)),
            b1_channels=int(match.group(2)),
            act=nn.ReLU(),
            fcl_channels=256)
    elif match.group(3) == 'mish':
        return ResNetAddSigmoid(
            b1_layer=int(match.group(1)),
            b1_channels=int(match.group(2)),
            act=Mish(),
            fcl_channels=256)
    elif match.group(3) == 'swish':
        return ResNetAddSigmoid(
            b1_layer=int(match.group(1)),
            b1_channels=int(match.group(2)),
            act=Swish(),
            fcl_channels=256)
    elif match.group(3) == 'tanhexp':
        return ResNetAddSigmoid(
            b1_layer=int(match.group(1)),
            b1_channels=int(match.group(2)),
            act=TanhExp(),
            fcl_channels=256)
    else:
        return None
