import torch
import torch.nn as nn
from restormer.models.upsampling import UpsamplingBlock
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Basic Residual Block with two 3x3 Convs"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)  # skip connection

class SRNetwork(nn.Module):
    def __init__(self, in_channels=32, out_channels=3, num_res_blocks=5, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
        self.entry = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])
        self.upsampling = UpsamplingBlock(64, scale_factor=scale_factor)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.residual_scale = nn.Parameter(torch.ones(1))

    def forward(self, x, bicubic=None):
        x = self.entry(x)
        x = self.res_blocks(x)
        x = self.upsampling(x)
        x = self.final_conv(x)
        if bicubic is not None:
            if bicubic.shape[2:] != x.shape[2:]:
                bicubic = F.interpolate(bicubic, size=x.shape[2:], mode='bicubic', align_corners=False)
            x = x + self.residual_scale * bicubic
        return x

class IntegratedGhostSR(nn.Module):
    """
    Integrated model combining GhostNet feature extraction with SR reconstruction
    """
    def __init__(self, ghostnet, sr_network):
        super().__init__()
        self.ghostnet = ghostnet
        self.sr_network = sr_network

    def forward(self, x, bicubic=None):
        features = self.ghostnet(x)
        return self.sr_network(features, bicubic)