"""Separate V3.1 candidates; no modification to the published V3 architecture."""
import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = {'a': (24, [1]*6), 'b': (32, [1]*6), 'c': (32, [1,1,2,2,1,1])}

def receptive_field(dilations):
    return 1 + 2*3 + 4*sum(dilations)  # head, fuse, tail + two convolutions/block

class Block(nn.Module):
    def __init__(self, channels, dilation, scale):
        super().__init__()
        self.scale = scale
        self.body = nn.Sequential(nn.Conv2d(channels,channels,3,padding=dilation,dilation=dilation),
                                  nn.LeakyReLU(.1),nn.Conv2d(channels,channels,3,padding=dilation,dilation=dilation))
    def forward(self, x):
        return x + self.scale*self.body(x)

class AdoreLiteV31(nn.Module):
    def __init__(self, variant='b', residual_scale=.5, base='bilinear'):
        super().__init__()
        channels, dilations = VARIANTS[variant]
        if base not in ('bilinear','bicubic'): raise ValueError('Invalid base interpolation')
        self.base = base
        self.head = nn.Sequential(nn.Conv2d(3,channels,3,padding=1),nn.LeakyReLU(.1))
        self.blocks = nn.Sequential(*(Block(channels,d,residual_scale) for d in dilations))
        self.fuse = nn.Conv2d(channels,channels,3,padding=1)
        self.tail = nn.Conv2d(channels,12,3,padding=1)
        nn.init.zeros_(self.tail.weight); nn.init.zeros_(self.tail.bias)
    def forward(self,x):
        h = self.head(x)
        h = h + self.fuse(self.blocks(h))
        return F.interpolate(x,scale_factor=2,mode=self.base,align_corners=False)+F.pixel_shuffle(self.tail(h),2)
