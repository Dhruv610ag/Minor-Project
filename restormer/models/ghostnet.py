import torch
import torch.nn as nn
import torch.nn.functional as F

class GhostModule(nn.Module):
    """
    Ghost Module as described in GhostNet paper.
    Generates more features from cheap operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_kernel_size=3, stride=1, relu=True):
        super().__init__()
        init_channels = int(out_channels / ratio)
        new_channels = out_channels - init_channels

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_kernel_size, 1, dw_kernel_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out

# In your GhostNet implementation, remove the upsampling and output layers:
class GhostNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Adjust input channels to N_frames * 3 (e.g., 3 frames -> 9 channels)
        self.input_conv = nn.Conv2d(9, 32, 3, padding=1)
        self.bottleneck1 = GhostBottleneck(32, 48, 64, stride=1)
        self.bottleneck2 = GhostBottleneck(64, 64, 64, stride=1)
        self.bottleneck3 = GhostBottleneck(64, 48, 32, stride=1)  # Output: 32 channels

    def forward(self, x):
        x = self.input_conv(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        return x  # [B, 32, H, W]
    
class GhostBottleneck(nn.Module):
    """
    Ghost Bottleneck block combining GhostModule with depthwise conv.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dw_kernel_size=3, stride=1):
        super().__init__()
        self.ghost1 = GhostModule(in_channels, hidden_channels, relu=True)
        self.dw_conv = nn.Conv2d(hidden_channels, hidden_channels, dw_kernel_size, stride, dw_kernel_size // 2, groups=hidden_channels, bias=False)
        self.dw_bn = nn.BatchNorm2d(hidden_channels)
        self.ghost2 = GhostModule(hidden_channels, out_channels, relu=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1, stride, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.ghost1(x)
        out = self.dw_conv(out)
        out = self.dw_bn(out)
        out = self.ghost2(out)
        res = self.shortcut(x) if len(self.shortcut) > 0 else x
        return self.relu(out + res)

class GhostNetStudentSR(nn.Module):
    def __init__(self, scale_factor=4):
        super().__init__()
        self.feature_extractor = GhostNetFeatureExtractor()
        # keep SR network separate (or call IntegratedGhostSR)
        self.scale_factor = scale_factor

    def forward(self, x):
        # expect x of shape [B, N*C, H, W] (concatenated frames)
        features = self.feature_extractor(x)
        return features  # to be passed to SR reconstruction network