"""
U-Net architecture with SatlasPretrain pretrained encoder.
"""
import logging

import torch
import torch.nn as nn

# Optional dependency - import at top level per cursorrules
try:
    import satlaspretrain_models
    SATLASPRETRAIN_AVAILABLE = True
except ImportError:
    SATLASPRETRAIN_AVAILABLE = False
    satlaspretrain_models = None  # Placeholder to avoid NameError

logger = logging.getLogger(__name__)


class InputAdapter(nn.Module):
    """
    Adapts 5-channel input (RGB + DEM + Slope) to 3-channel for pretrained encoder.

    Uses learnable fusion to preserve information from all channels.
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 3):
        """
        Initialize input adapter.

        Args:
            in_channels: Number of input channels (default: 5 for RGB+DEM+Slope)
            out_channels: Number of output channels (default: 3 for RGB)
        """
        super().__init__()
        # Process RGB and DEM+Slope separately, then fuse
        self.rgb_conv = nn.Conv2d(3, 3, kernel_size=1)
        self.dem_slope_conv = nn.Conv2d(2, 3, kernel_size=1)
        self.fusion = nn.Conv2d(6, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, 5, H, W] (RGB + DEM + Slope)

        Returns:
            Adapted tensor [B, 3, H, W]
        """
        # Split into RGB and DEM+Slope
        rgb = x[:, :3, :, :]  # First 3 channels (RGB)
        dem_slope = x[:, 3:, :, :]  # Last 2 channels (DEM + Slope)

        # Process separately
        rgb_features = self.rgb_conv(rgb)
        dem_slope_features = self.dem_slope_conv(dem_slope)

        # Concatenate and fuse
        fused = torch.cat([rgb_features, dem_slope_features], dim=1)
        output = self.fusion(fused)
        output = self.bn(output)
        output = self.relu(output)

        return output


class SatlasPretrainUNet(nn.Module):
    """
    U-Net architecture with SatlasPretrain pretrained encoder.

    Supports ResNet50, ResNet152, Swin-v2-Base, and Swin-v2-Tiny encoders.
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 1,
        encoder_name: str = "resnet50",
        pretrained: bool = True,
        freeze_encoder: bool = True,
        decoder_dropout: float = 0.2,
    ):
        """
        Initialize SatlasPretrain U-Net.

        Args:
            in_channels: Number of input channels (default: 5 for RGB+DEM+Slope)
            out_channels: Number of output channels (default: 1 for regression)
            encoder_name: Encoder name - "resnet50", "resnet152", "swin_v2_base", "swin_v2_tiny"
            pretrained: Whether to use pretrained weights (default: True)
            freeze_encoder: Whether to freeze encoder weights initially (default: True)
            decoder_dropout: Dropout probability in decoder (default: 0.2)
        """
        super().__init__()

        self.encoder_name = encoder_name
        self.pretrained = pretrained
        self.freeze_encoder = freeze_encoder

        # Input adapter for 5-channel input
        self.input_adapter = InputAdapter(in_channels=in_channels, out_channels=3)

        # Load pretrained encoder
        self.encoder = self._load_encoder(encoder_name, pretrained)

        # Get encoder feature dimensions
        encoder_dims = self._get_encoder_dims(encoder_name)

        # Build decoder
        self.decoder = self._build_decoder(encoder_dims, decoder_dropout)

        # Output head
        self.final_conv = nn.Conv2d(encoder_dims[0], out_channels, kernel_size=1)

        # Freeze encoder if requested
        if freeze_encoder:
            self._freeze_encoder()

    def _load_encoder(self, encoder_name: str, pretrained: bool) -> nn.Module:
        """Load pretrained encoder from SatlasPretrain."""
        if not SATLASPRETRAIN_AVAILABLE:
            raise ImportError(
                "satlaspretrain-models is not installed. "
                "Install it with: pip install satlaspretrain-models"
            )

        weights_manager = satlaspretrain_models.Weights()

        # Map encoder names to model identifiers
        # For aerial imagery (0.2m/pixel), use Aerial models
        # For ResNet, fall back to Sentinel-2 models (closest match)
        model_identifiers = {
            "resnet50": "Sentinel2_Resnet50_SI_RGB",  # Sentinel-2 RGB (closest to aerial RGB)
            "resnet152": "Sentinel2_Resnet152_SI_RGB",
            "swin_v2_base": "Aerial_SwinB_SI",  # Aerial model (best for high-res)
            "swin_v2_tiny": "Aerial_SwinT_SI",  # Aerial model (best for high-res)
        }

        if encoder_name not in model_identifiers:
            raise ValueError(
                f"Unknown encoder: {encoder_name}. "
                f"Supported: {list(model_identifiers.keys())}"
            )

        model_id = model_identifiers[encoder_name]
        logger.info(f"Loading SatlasPretrain encoder: {model_id} (pretrained={pretrained})")

        # Load backbone only (no FPN, no head)
        encoder = weights_manager.get_pretrained_model(model_id, fpn=False)

        return encoder

    def _get_encoder_dims(self, encoder_name: str) -> list[int]:
        """
        Get encoder feature dimensions for decoder construction.

        Returns list of channel dimensions at each scale.
        """
        if encoder_name in ["resnet50", "resnet152"]:
            # ResNet outputs: [256, 512, 1024, 2048] for ResNet50
            # ResNet outputs: [256, 512, 1024, 2048] for ResNet152
            return [256, 512, 1024, 2048]
        elif encoder_name == "swin_v2_base":
            # Swin-v2-Base outputs: [128, 256, 512, 1024]
            return [128, 256, 512, 1024]
        elif encoder_name == "swin_v2_tiny":
            # Swin-v2-Tiny outputs: [96, 192, 384, 768]
            return [96, 192, 384, 768]
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")

    def _build_decoder(self, encoder_dims: list[int], dropout: float) -> nn.ModuleDict:
        """Build U-Net decoder matching encoder dimensions."""
        from src.models.architectures import DoubleConv

        decoder = nn.ModuleDict()

        # Bottleneck (from deepest encoder features)
        decoder["bottleneck"] = DoubleConv(encoder_dims[3], encoder_dims[3] * 2, dropout=0.0)

        # Decoder blocks (upsampling + skip connections)
        # Reverse order: from deepest to shallowest
        decoder["up4"] = nn.ConvTranspose2d(encoder_dims[3] * 2, encoder_dims[3], kernel_size=2, stride=2)
        decoder["dec4"] = DoubleConv(encoder_dims[3] + encoder_dims[3], encoder_dims[3], dropout=dropout)

        decoder["up3"] = nn.ConvTranspose2d(encoder_dims[3], encoder_dims[2], kernel_size=2, stride=2)
        decoder["dec3"] = DoubleConv(encoder_dims[2] + encoder_dims[2], encoder_dims[2], dropout=dropout)

        decoder["up2"] = nn.ConvTranspose2d(encoder_dims[2], encoder_dims[1], kernel_size=2, stride=2)
        decoder["dec2"] = DoubleConv(encoder_dims[1] + encoder_dims[1], encoder_dims[1], dropout=dropout)

        decoder["up1"] = nn.ConvTranspose2d(encoder_dims[1], encoder_dims[0], kernel_size=2, stride=2)
        decoder["dec1"] = DoubleConv(encoder_dims[0] + encoder_dims[0], encoder_dims[0], dropout=dropout)

        return decoder

    def _freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen (parameters require_grad=False)")

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("Encoder unfrozen (parameters require_grad=True)")

    def _match_size(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Match spatial dimensions of x to target using interpolation.

        Args:
            x: Tensor to resize [B, C, H, W]
            target: Target tensor [B, C, H', W']

        Returns:
            Resized tensor matching target spatial dimensions
        """
        if x.shape[2:] != target.shape[2:]:
            x = torch.nn.functional.interpolate(
                x, size=target.shape[2:], mode='bilinear', align_corners=False
            )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, 5, H, W] (RGB + DEM + Slope)

        Returns:
            Output tensor [B, 1, H, W] matching input spatial dimensions
        """
        # Store original input size for final upsampling
        input_size = x.shape[2:]

        # Adapt input to 3 channels
        x = self.input_adapter(x)

        # Encoder forward pass
        # SatlasPretrain encoders return list of feature maps at different scales
        encoder_features = self.encoder(x)

        # Extract features at 4 scales (for U-Net skip connections)
        # encoder_features is a list: [feat1, feat2, feat3, feat4]
        enc1 = encoder_features[0]  # Shallowest (largest spatial size)
        enc2 = encoder_features[1]
        enc3 = encoder_features[2]
        enc4 = encoder_features[3]  # Deepest (smallest spatial size)

        # Bottleneck
        bottleneck = self.decoder["bottleneck"](enc4)

        # Decoder with skip connections
        # Match sizes before concatenation
        dec4 = self.decoder["up4"](bottleneck)
        dec4 = self._match_size(dec4, enc4)  # Match decoder to encoder size
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder["dec4"](dec4)

        dec3 = self.decoder["up3"](dec4)
        dec3 = self._match_size(dec3, enc3)  # Match decoder to encoder size
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder["dec3"](dec3)

        dec2 = self.decoder["up2"](dec3)
        dec2 = self._match_size(dec2, enc2)  # Match decoder to encoder size
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder["dec2"](dec2)

        dec1 = self.decoder["up1"](dec2)
        dec1 = self._match_size(dec1, enc1)  # Match decoder to encoder size
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder["dec1"](dec1)

        # Output
        output = self.final_conv(dec1)

        # Ensure output matches input spatial dimensions
        if output.shape[2:] != input_size:
            output = torch.nn.functional.interpolate(
                output, size=input_size, mode='bilinear', align_corners=False
            )

        return output
