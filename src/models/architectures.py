"""
U-Net architectures for semantic segmentation.
"""

import torch
import torch.nn as nn


def _bound_proximity(
    x: torch.Tensor,
    max_val: float,
    activation: str,
    temperature: float,
) -> torch.Tensor:
    if activation == "clamp":
        return x.clamp(0.0, max_val)
    if activation == "sigmoid":
        return torch.sigmoid(x) * max_val
    if activation == "sigmoid_steep":
        return torch.sigmoid(x / max(temperature, 1e-6)) * max_val
    return x.clamp(0.0, max_val)


class DoubleConv(nn.Module):
    """Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> Dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        """
        Initialize DoubleConv block.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            dropout: Dropout probability (default: 0.0 = disabled)
        """
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Classic U-Net architecture for semantic segmentation.

    Simple baseline model without pretrained encoder.
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 1,
        base_channels: int = 64,
        dropout: float = 0.2,
        proximity_max: float = 0,
        output_activation: str = "clamp",
        sigmoid_temperature: float = 0.3,
    ):
        """
        Initialize U-Net.

        Args:
            in_channels: Number of input channels (default: 5 for RGB+DEM+Slope)
            out_channels: Number of output channels (default: 1 for regression)
            base_channels: Base number of channels (default: 64)
            dropout: Dropout probability for regularization (default: 0.2)
            proximity_max: If > 0, bound output to [0, proximity_max] (default: 0 = no bound)
            output_activation: "clamp" | "sigmoid" | "sigmoid_steep". clamp = no saturation; sigmoid_steep = larger gradients near 0/20.
            sigmoid_temperature: For sigmoid_steep, scale raw logits (default 0.3 = steeper).
        """
        super().__init__()
        self.proximity_max = proximity_max
        self.output_activation = (output_activation or "clamp").lower()
        self.sigmoid_temperature = sigmoid_temperature

        # Encoder (contracting path) - no dropout (keep features)
        self.enc1 = DoubleConv(in_channels, base_channels, dropout=0.0)
        self.enc2 = DoubleConv(base_channels, base_channels * 2, dropout=0.0)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4, dropout=0.0)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8, dropout=0.0)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        # Decoder (expanding path) - add dropout for regularization
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8, dropout=dropout)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4, dropout=dropout)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2, dropout=dropout)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels, dropout=dropout)

        # Output head
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder with skip connections
        dec4 = self.up4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.up3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        output = self.final_conv(dec1)
        if self.proximity_max > 0:
            output = _bound_proximity(output, self.proximity_max, self.output_activation, self.sigmoid_temperature)
        return output
