#!/usr/bin/env python3
"""
Test script to verify model factory and architecture switching.
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from src.models.factory import create_model
from src.utils.path_utils import get_project_root

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_unet_baseline():
    """Test baseline UNet creation."""
    logger.info("=" * 60)
    logger.info("Testing baseline UNet")
    logger.info("=" * 60)
    
    config = {
        "architecture": "unet",
        "in_channels": 5,
        "out_channels": 1,
        "base_channels": 64,
        "dropout": 0.2,
    }
    
    model = create_model(config)
    
    # Test forward pass
    x = torch.randn(1, 5, 256, 256)
    output = model(x)
    
    logger.info("✓ UNet created successfully")
    logger.info(f"  Input shape: {x.shape}")
    logger.info(f"  Output shape: {output.shape}")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    assert output.shape == (1, 1, 256, 256), f"Expected output shape (1, 1, 256, 256), got {output.shape}"
    logger.info("✓ Forward pass successful\n")
    
    return model


def test_satlaspretrain_unet():
    """Test SatlasPretrain U-Net creation."""
    logger.info("=" * 60)
    logger.info("Testing SatlasPretrain U-Net")
    logger.info("=" * 60)
    
    try:
        config = {
            "architecture": "satlaspretrain_unet",
            "in_channels": 5,
            "out_channels": 1,
            "encoder": {
                "name": "resnet50",
                "pretrained": True,
                "freeze_encoder": True,
                "unfreeze_after_epoch": 10,
            },
            "decoder_dropout": 0.2,
        }
        
        model = create_model(config)
        
        # Test forward pass
        x = torch.randn(1, 5, 256, 256)
        output = model(x)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        
        logger.info("✓ SatlasPretrain U-Net created successfully")
        logger.info(f"  Input shape: {x.shape}")
        logger.info(f"  Output shape: {output.shape}")
        logger.info(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Frozen parameters: {frozen_params:,}")
        logger.info(f"  Encoder frozen: {not any(p.requires_grad for p in model.encoder.parameters())}")
        
        assert output.shape == (1, 1, 256, 256), f"Expected output shape (1, 1, 256, 256), got {output.shape}"
        logger.info("✓ Forward pass successful\n")
        
        # Test encoder unfreezing
        logger.info("Testing encoder unfreezing...")
        model.unfreeze_encoder()
        trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Trainable parameters after unfreezing: {trainable_after:,}")
        logger.info(f"  Encoder unfrozen: {any(p.requires_grad for p in model.encoder.parameters())}")
        logger.info("✓ Encoder unfreezing successful\n")
        
        return model
        
    except ImportError as e:
        logger.warning(f"⚠ SatlasPretrain not available: {e}")
        logger.warning("  Install with: pip install satlaspretrain-models")
        logger.warning("  Skipping SatlasPretrain test\n")
        return None


def test_config_file():
    """Test loading model from config file."""
    logger.info("=" * 60)
    logger.info("Testing config file loading")
    logger.info("=" * 60)
    
    project_root = get_project_root(__file__)
    config_path = project_root / "configs" / "training_config.yaml"
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Test UNet from config
    model_config = config["model"].copy()
    model_config["architecture"] = "unet"  # Ensure we test UNet
    model = create_model(model_config)
    
    logger.info("✓ Model loaded from config file")
    logger.info(f"  Architecture: {model_config.get('architecture', 'unet')}")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")


def main():
    """Run all tests."""
    logger.info("Testing Model Factory Implementation\n")
    
    # Test 1: Baseline UNet
    test_unet_baseline()
    
    # Test 2: SatlasPretrain U-Net (if available)
    test_satlaspretrain_unet()
    
    # Test 3: Config file loading
    test_config_file()
    
    logger.info("=" * 60)
    logger.info("All tests completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
