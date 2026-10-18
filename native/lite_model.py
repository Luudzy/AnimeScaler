"""Small experimental student. Trained against Adore outputs, never random in production."""
import torch
from torch import nn
from torch.nn import functional as F


class AdoreLite(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3,channels,3,padding=1), nn.LeakyReLU(.1),
                                  nn.Conv2d(channels,channels,3,padding=1), nn.LeakyReLU(.1),
                                  nn.Conv2d(channels,12,3,padding=1), nn.PixelShuffle(2))
        nn.init.zeros_(self.body[4].weight)
        nn.init.zeros_(self.body[4].bias)

    def forward(self, value):
        return F.interpolate(value, scale_factor=2, mode='bilinear', align_corners=False) + self.body(value)
