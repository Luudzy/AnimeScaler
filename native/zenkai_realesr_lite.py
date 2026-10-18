"""Compact SRVGG-inspired 2x candidate for the Zenkai smoke gate.

This is deliberately a separate experimental model. It borrows the useful
AnimeVideo ideas (PReLU body and explicit reconstruction residual) without
copying its weights or replacing any existing Zenkai/Adore artifact.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class SRVGGResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, value):
        return value + self.body(value)


class ZenkaiRealESRLite(nn.Module):
    def __init__(self, channels=32, blocks=6, scale=2):
        super().__init__()
        if channels < 4 or channels % 4:
            raise ValueError('channels must be a positive multiple of four')
        if blocks != 6:
            raise ValueError('Zenkai-RealESR Lite uses exactly six blocks')
        if scale != 2:
            raise ValueError('Zenkai-RealESR Lite supports native x2 only')
        self.config = {
            'family': 'Zenkai',
            'id': 'zenkai-realesr-lite-v1',
            'architecture': 'SRVGG-inspired PReLU',
            'channels': channels,
            'blocks': blocks,
            'nativeScale': scale,
            'legacy': False,
            'residual': 'bilinear x2 plus zero-initialized PixelShuffle residual',
        }
        self.head = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.PReLU(channels),
        )
        self.blocks = nn.Sequential(*(SRVGGResidualBlock(channels) for _ in range(blocks)))
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.PReLU(channels),
        )
        self.tail = nn.Conv2d(channels, 12, 3, padding=1)
        # Step-zero output is exactly the deterministic bilinear baseline.
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    @staticmethod
    def _validate_input(value):
        if value.ndim != 4 or value.shape[1] != 3:
            raise ValueError('Expected NCHW RGB input')
        height, width = value.shape[-2:]
        if min(height, width) < 4 or height % 4 or width % 4:
            raise ValueError('Input dimensions must be at least 4 and divisible by four')

    def forward(self, value):
        self._validate_input(value)
        features = self.head(value)
        features = self.blocks(features)
        residual = F.pixel_shuffle(self.tail(self.fuse(features)), 2)
        base = F.interpolate(value, scale_factor=2, mode='bilinear', align_corners=False)
        return base + residual

    def forward_to(self, value, output_size):
        self._validate_input(value)
        size = tuple(int(v) for v in output_size)
        if len(size) != 2 or min(size) < 2:
            raise ValueError('Output dimensions must be positive')
        output = self(value)
        return output if tuple(output.shape[-2:]) == size else F.interpolate(
            output, size=size, mode='bilinear', align_corners=False)

    def deploy_copy(self):
        # Keep a named hook for the common training/export pipeline. PReLU and
        # residual blocks are already deploy-safe; only clone and mark metadata.
        result = copy.deepcopy(self)
        result.config = {**self.config, 'deploy': True}
        return result


class ZenkaiAnimeSRLite(nn.Module):
    """Multi-scale anime-oriented variant inspired by AnimeSR's scale pyramid.

    All expensive convolutions remain in LR or reduced-resolution feature maps;
    the final reconstruction is still a zero-initialized x2 residual.
    """
    def __init__(self, channels=32, blocks=6, scale=2):
        super().__init__()
        if channels < 4 or channels % 4 or blocks != 6 or scale != 2:
            raise ValueError('Zenkai-AnimeSR Lite uses channels multiple of four, six blocks and native x2')
        self.config = {'family':'Zenkai', 'id':'zenkai-animesr-lite-v1',
                       'architecture':'AnimeSR-inspired multi-scale PReLU',
                       'channels':channels, 'blocks':blocks, 'nativeScale':scale,
                       'legacy':False, 'residual':'bilinear x2 plus zero-initialized PixelShuffle residual'}
        self.head = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1), nn.PReLU(channels))
        self.full = nn.Sequential(*(SRVGGResidualBlock(channels) for _ in range(3)))
        self.half_down = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        self.half_blocks = nn.Sequential(*(SRVGGResidualBlock(channels) for _ in range(2)))
        self.quarter_down = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        self.quarter = SRVGGResidualBlock(channels)
        self.fuse = nn.Sequential(nn.Conv2d(channels * 3, channels, 1), nn.PReLU(channels),
                                  nn.Conv2d(channels, channels, 3, padding=1), nn.PReLU(channels))
        self.tail = nn.Conv2d(channels, 12, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    @staticmethod
    def _validate_input(value):
        ZenkaiRealESRLite._validate_input(value)

    def forward(self, value):
        self._validate_input(value)
        base_features = self.head(value)
        full = self.full(base_features)
        half = self.half_blocks(self.half_down(base_features))
        quarter = self.quarter(self.quarter_down(half))
        half_up = F.interpolate(half, size=full.shape[-2:], mode='bilinear', align_corners=False)
        quarter_up = F.interpolate(quarter, size=full.shape[-2:], mode='bilinear', align_corners=False)
        residual = F.pixel_shuffle(self.tail(self.fuse(torch.cat((full, half_up, quarter_up), dim=1))), 2)
        return F.interpolate(value, scale_factor=2, mode='bilinear', align_corners=False) + residual

    def deploy_copy(self):
        result = copy.deepcopy(self)
        result.config = {**self.config, 'deploy': True}
        return result
