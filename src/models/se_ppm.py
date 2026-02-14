"""
Squeeze-and-Excitation (SE) and Pyramid Pooling Module (PPM) for encoder-decoder models.

Used between encoder and decoder (e.g. on deepest feature enc4) for channel reweighting (SE)
and multi-scale context (PPM). Reference: Gully-ERFNet; PSPNet; Yu et al. IEEE JSTARS 2018.
"""

import torch
import torch.nn as nn


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation block: channel attention via global pooling and two FC layers.

    Input and output shape: [B, C, H, W].
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction) if channels >= reduction else max(1, channels // 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class PyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module: multi-scale context via adaptive pooling and concat.

    Input and output shape: [B, C, H, W]. Uses bins (e.g. 1,2,3,6) for pool sizes.
    """

    def __init__(self, channels: int, bins: tuple = (1, 2, 3, 6)):
        super().__init__()
        self.bins = bins
        n = len(bins)
        branch_channels = max(1, channels // n)
        self.branches = nn.ModuleList()
        for b in bins:
            pool_size = (b, b)
            self.branches.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size),
                    nn.Conv2d(channels, branch_channels, kernel_size=1),
                    nn.ReLU(inplace=True),
                )
            )
        self.fuse = nn.Conv2d(channels + n * branch_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        out = [x]
        for branch in self.branches:
            pooled = branch(x)
            out.append(
                nn.functional.interpolate(
                    pooled, size=(h, w), mode="bilinear", align_corners=False
                )
            )
        out = torch.cat(out, dim=1)
        return self.fuse(out)
