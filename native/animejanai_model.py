"""Exact SuperUltraCompact 2x topology from the shipped ONNX model."""
import torch
from torch import nn
from torch.nn import functional as F


class AnimeJaNai(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for index in range(9):
            layers.extend([nn.Conv2d(3 if index == 0 else 24, 24, 3, padding=1), nn.PReLU(24)])
        layers.append(nn.Conv2d(24, 12, 3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, value):
        return (F.pixel_shuffle(self.body(value), 2) + F.interpolate(value, scale_factor=2, mode='nearest')).clamp(0, 1)
