import torch
import torch.nn as nn
# REMOVED: from restormer.models.upsampling import UpsamplingBlock
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

class EnhancementNetwork(nn.Module):  # CHANGED: Renamed from SRNetwork
    def __init__(self, in_channels=32, out_channels=3, num_res_blocks=5):
        super().__init__()
        # REMOVED: scale_factor parameter
        self.entry = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])
        # REMOVED: upsampling block
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.residual_scale = nn.Parameter(torch.ones(1))

    def forward(self, x, bicubic=None):
        x = self.entry(x)
        x = self.res_blocks(x)
        # REMOVED: upsampling step
        x = self.final_conv(x)
        if bicubic is not None:
            # bicubic should be same resolution now (no upscaling needed)
            if bicubic.shape[2:] != x.shape[2:]:
                bicubic = F.interpolate(bicubic, size=x.shape[2:], mode='bicubic', align_corners=False)
            x = x + self.residual_scale * bicubic
        return x

class IntegratedGhostEnhancer(nn.Module):  # CHANGED: Renamed from IntegratedGhostSR
    """
    Integrated model combining GhostNet feature extraction with enhancement
    """
    def __init__(self, ghostnet, enhance_network):  # CHANGED: parameter name
        super().__init__()
        self.ghostnet = ghostnet
        self.enhance_network = enhance_network  # CHANGED: variable name

    def forward(self, x, bicubic=None):
        features = self.ghostnet(x)
        return self.enhance_network(features, bicubic)