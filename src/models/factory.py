"""
Model factory for creating different architectures.
"""
import logging
from typing import Dict, Any
import torch.nn as nn

from src.models.architectures import UNet

logger = logging.getLogger(__name__)


def create_model(config: Dict[str, Any]) -> nn.Module:
    """
    Create model based on configuration.

    Args:
        config: Model configuration dictionary

    Returns:
        Initialized model

    Raises:
        ValueError: If architecture is unknown or configuration is invalid
    """
    architecture = config.get("architecture", "unet").lower()

    # Validate architecture
    if architecture not in ["unet", "satlaspretrain_unet"]:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Supported: 'unet', 'satlaspretrain_unet'"
        )

    if architecture == "unet":
        logger.info("Creating baseline UNet architecture")
        return UNet(
            in_channels=config.get("in_channels", 5),
            out_channels=config.get("out_channels", 1),
            base_channels=config.get("base_channels", 64),
            dropout=config.get("dropout", 0.2),
            proximity_max=float(config.get("proximity_max", 20)),
            output_activation=config.get("output_activation", "clamp"),
            sigmoid_temperature=float(config.get("sigmoid_temperature", 0.3)),
        )
    elif architecture == "satlaspretrain_unet":
        logger.info("Creating SatlasPretrain U-Net architecture")
        try:
            from src.models.satlaspretrain_unet import SatlasPretrainUNet
        except ImportError as e:
            logger.error(f"Failed to import SatlasPretrainUNet: {e}")
            logger.error("Make sure satlaspretrain-models is installed: pip install satlaspretrain-models")
            raise

        encoder_config = config.get("encoder", {})
        encoder_name = encoder_config.get("name", "resnet50")

        # Validate encoder name
        valid_encoders = ["resnet50", "resnet152", "swin_v2_base", "swin_v2_tiny"]
        if encoder_name not in valid_encoders:
            raise ValueError(
                f"Unknown encoder name: {encoder_name}. "
                f"Supported: {valid_encoders}"
            )

        ppm_bins_raw = config.get("ppm_bins", (1, 2, 3, 6))
        ppm_bins = tuple(ppm_bins_raw) if not isinstance(ppm_bins_raw, tuple) else ppm_bins_raw

        return SatlasPretrainUNet(
            in_channels=config.get("in_channels", 5),
            out_channels=config.get("out_channels", 1),
            encoder_name=encoder_name,
            pretrained=encoder_config.get("pretrained", True),
            freeze_encoder=encoder_config.get("freeze_encoder", True),
            decoder_dropout=config.get("decoder_dropout", 0.2),
            use_se=config.get("use_se", False),
            use_ppm=config.get("use_ppm", False),
            ppm_bins=ppm_bins,
            se_reduction=config.get("se_reduction", 16),
            proximity_max=float(config.get("proximity_max", 20)),
            output_activation=config.get("output_activation", "clamp"),
            sigmoid_temperature=float(config.get("sigmoid_temperature", 0.3)),
        )
