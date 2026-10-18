"""Small residual SR candidate; kept separate from deployed Lite V2."""
import torch
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self,channels,scale=.2):
        super().__init__();self.scale=scale
        self.body=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1),nn.LeakyReLU(.1),nn.Conv2d(channels,channels,3,padding=1))
    def forward(self,x):return x+self.body(x)*self.scale

class AdoreLiteV3(nn.Module):
    def __init__(self,channels=24,blocks=4,residual_scale=.2):
        super().__init__()
        self.head=nn.Sequential(nn.Conv2d(3,channels,3,padding=1),nn.LeakyReLU(.1))
        self.blocks=nn.Sequential(*(ResidualBlock(channels,residual_scale) for _ in range(blocks)))
        self.fuse=nn.Conv2d(channels,channels,3,padding=1)
        self.tail=nn.Conv2d(channels,12,3,padding=1)
        nn.init.zeros_(self.tail.weight);nn.init.zeros_(self.tail.bias)
    def forward(self,x):
        h=self.head(x);h=h+self.fuse(self.blocks(h))
        return F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)+F.pixel_shuffle(self.tail(h),2)
