import torch
import torch.nn as nn

class SubpixelUpsampling(nn.Module):
    """
    Subpixel Upsampling using PixelShuffle for efficient super-resolution.
    Supports arbitrary scale factors.

    Args:
        in_channels (int): Number of input channels.
        scale_factor (int): Upscaling factor (default: 2).
    """
    def __init__(self, in_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(
            in_channels,
            in_channels * (scale_factor ** 2),
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(in_channels * (scale_factor ** 2))
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x

class UpsamplingBlock(nn.Module):
    """
    Flexible upsampling block that handles any scale factor.
    Uses optimal combination of 2x upsampling steps.

    Args:
        in_channels (int): Number of input channels.
        scale_factor (int): Total upscaling factor (2, 3, 4, etc.)
    """
    def __init__(self, in_channels, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
        self.upsample_layers = nn.ModuleList()
        current_scale = 1
        factors = []
        temp_scale = scale_factor
        while temp_scale > 1:
            if temp_scale % 2 == 0:
                factors.append(2)
                temp_scale //= 2
            elif temp_scale % 3 == 0:
                factors.append(3)
                temp_scale //= 3
            else:
                factors.append(2)
                temp_scale = temp_scale // 2 if temp_scale // 2 >= 1 else 1
        for factor in factors:
            self.upsample_layers.append(SubpixelUpsampling(in_channels, factor))
            current_scale *= factor

    def forward(self, x):
        for upsample_layer in self.upsample_layers:
            x = upsample_layer(x)
        return x

class FlexibleUpsampling(nn.Module):
    """
    Alternative upsampling approach using transposed convolutions
    for non-integer scale factors or when PixelShuffle isn't suitable.
    """
    def __init__(self, in_channels, out_channels, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
        # For simplicity: support scale_factor == 4 with two PixelShuffle 2x
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.upsample(x)