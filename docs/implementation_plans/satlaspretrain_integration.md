# Implementation Plan: SatlasPretrain Models Integration

## Overview

This plan outlines the implementation of SatlasPretrain pretrained encoders for the U-Net architecture, with a flexible system to easily switch between different model architectures (baseline UNet, SatlasPretrain, and future options).

## Goals

1. **Integrate SatlasPretrain models** as pretrained encoder option
2. **Maintain backward compatibility** with existing baseline UNet
3. **Enable easy architecture switching** via configuration
4. **Handle 5-channel input** (RGB + DEM + Slope) properly
5. **Support encoder freezing/unfreezing** for transfer learning

## Architecture Design

### 1. Model Factory Pattern

Create a centralized model factory that handles architecture instantiation:

```
src/models/
├── __init__.py
├── architectures.py          # Existing UNet + new factory
├── factory.py                # NEW: Model factory for architecture switching
├── satlaspretrain_unet.py    # NEW: U-Net with SatlasPretrain encoder
└── losses.py                 # Existing
```

### 2. Configuration Structure

Extend `configs/training_config.yaml` to support architecture selection:

```yaml
model:
  architecture: "satlaspretrain_unet"  # Options: "unet", "satlaspretrain_unet", future options
  in_channels: 5
  out_channels: 1

  # Architecture-specific parameters
  encoder:
    name: "resnet50"  # Options: "resnet50", "resnet152", "swin_v2_base", "swin_v2_tiny"
    pretrained: true
    freeze_encoder: true  # Freeze encoder initially, unfreeze after convergence
    unfreeze_after_epoch: 10  # Unfreeze encoder after this epoch (0 = never unfreeze)

  # UNet-specific (only used if architecture == "unet")
  base_channels: 64
  dropout: 0.2

  # SatlasPretrain-specific (only used if architecture == "satlaspretrain_unet")
  decoder_dropout: 0.2  # Dropout in decoder only
```

## Implementation Steps

### Step 1: Install Dependencies

**File**: `pyproject.toml`

Add dependencies:
```toml
[project]
dependencies = [
    # ... existing dependencies ...
    "satlaspretrain-models>=0.1.0",  # SatlasPretrain models
    "torchgeo>=0.5.0",  # Optional: for geospatial utilities
]
```

**Action**: Update `pyproject.toml` and run `poetry install`

### Step 2: Create Model Factory

**File**: `src/models/factory.py` (NEW)

```python
"""
Model factory for creating different architectures.
"""
from typing import Dict, Any
import torch.nn as nn
from src.models.architectures import UNet
from src.models.satlaspretrain_unet import SatlasPretrainUNet


def create_model(config: Dict[str, Any]) -> nn.Module:
    """
    Create model based on configuration.

    Args:
        config: Model configuration dictionary

    Returns:
        Initialized model
    """
    architecture = config.get("architecture", "unet").lower()

    if architecture == "unet":
        return UNet(
            in_channels=config.get("in_channels", 5),
            out_channels=config.get("out_channels", 1),
            base_channels=config.get("base_channels", 64),
            dropout=config.get("dropout", 0.2),
        )
    elif architecture == "satlaspretrain_unet":
        encoder_config = config.get("encoder", {})
        return SatlasPretrainUNet(
            in_channels=config.get("in_channels", 5),
            out_channels=config.get("out_channels", 1),
            encoder_name=encoder_config.get("name", "resnet50"),
            pretrained=encoder_config.get("pretrained", True),
            freeze_encoder=encoder_config.get("freeze_encoder", True),
            decoder_dropout=config.get("decoder_dropout", 0.2),
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
```

### Step 3: Implement SatlasPretrain U-Net

**File**: `src/models/satlaspretrain_unet.py` (NEW)

Key components:
1. **5-channel input adapter**: Adapt first conv layer to accept 5 channels
2. **SatlasPretrain encoder**: Load pretrained ResNet50/ResNet152/Swin-v2
3. **U-Net decoder**: Standard decoder with skip connections
4. **Encoder freezing**: Support freezing/unfreezing encoder during training

**Implementation approach**:
- Use `satlaspretrain_models` package to load pretrained encoders
- Extract encoder features at multiple scales (for skip connections)
- Build U-Net decoder that matches encoder feature dimensions
- Handle 5-channel input by adapting first layer

### Step 4: Update Training Script

**File**: `scripts/train_model.py`

**Changes**:
1. Replace direct `UNet()` instantiation with factory call
2. Add encoder unfreezing logic during training
3. Log encoder information to MLflow

**Key modifications**:
```python
# OLD:
from src.models.architectures import UNet
model = UNet(...)

# NEW:
from src.models.factory import create_model
model = create_model(config["model"])

# Add encoder unfreezing logic:
if hasattr(model, 'unfreeze_encoder') and config["model"]["encoder"].get("unfreeze_after_epoch", 0) > 0:
    # Unfreeze encoder after specified epoch
    ...
```

### Step 5: Update Configuration

**File**: `configs/training_config.yaml`

Add architecture selection and encoder configuration (see Step 2 for structure).

### Step 6: Handle 5-Channel Input

**Challenge**: SatlasPretrain models expect 3-channel RGB input, but we have 5 channels (RGB + DEM + Slope).

**Solution**: Create an input adapter layer:
```python
class InputAdapter(nn.Module):
    """Adapts 5-channel input to 3-channel for pretrained encoder."""
    def __init__(self, in_channels=5, out_channels=3):
        super().__init__()
        # Option 1: Simple projection
        self.adapter = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Option 2: Learnable fusion (better)
        # Process RGB and DEM+Slope separately, then fuse
        self.rgb_conv = nn.Conv2d(3, 3, kernel_size=1)
        self.dem_slope_conv = nn.Conv2d(2, 3, kernel_size=1)
        self.fusion = nn.Conv2d(6, 3, kernel_size=1)
```

**Recommendation**: Use Option 2 (learnable fusion) for better feature preservation.

### Step 7: Testing & Validation

1. **Unit tests**: Test model creation, forward pass, encoder freezing
2. **Integration test**: Run training for 1-2 epochs to verify
3. **Backward compatibility**: Ensure existing UNet still works
4. **MLflow logging**: Verify encoder info is logged correctly

## Technical Details

### SatlasPretrain Encoder Integration

**Available backbones**:
- `resnet50`: ResNet-50 (recommended for balance of performance/speed)
- `resnet152`: ResNet-152 (better performance, slower)
- `swin_v2_base`: Swin Transformer v2 Base
- `swin_v2_tiny`: Swin Transformer v2 Tiny (faster, less accurate)

**Loading pretrained weights**:
```python
from satlaspretrain_models import load_pretrained_model

# Load encoder only
encoder = load_pretrained_model(
    model_name="resnet50",
    task="classification",  # or "segmentation"
    pretrained=True
)
```

### Encoder Feature Extraction

SatlasPretrain encoders output features at multiple scales. We need to:
1. Extract features at 4 scales (for U-Net skip connections)
2. Match decoder dimensions to encoder feature dimensions
3. Handle different encoder architectures (ResNet vs Swin)

### Transfer Learning Strategy

**Phase 1: Frozen Encoder** (epochs 0-N)
- Freeze encoder weights
- Train only decoder and output head
- Lower learning rate for decoder (e.g., 0.001)

**Phase 2: Fine-tuning** (epochs N+)
- Unfreeze encoder
- Use lower learning rate for encoder (e.g., 0.0001)
- Continue training all layers

## Configuration Examples

### Example 1: SatlasPretrain ResNet50 (Recommended)
```yaml
model:
  architecture: "satlaspretrain_unet"
  in_channels: 5
  out_channels: 1
  encoder:
    name: "resnet50"
    pretrained: true
    freeze_encoder: true
    unfreeze_after_epoch: 10
  decoder_dropout: 0.2
```

### Example 2: Baseline UNet (Backward Compatible)
```yaml
model:
  architecture: "unet"
  in_channels: 5
  out_channels: 1
  base_channels: 64
  dropout: 0.2
```

### Example 3: SatlasPretrain Swin-v2 (Future)
```yaml
model:
  architecture: "satlaspretrain_unet"
  in_channels: 5
  out_channels: 1
  encoder:
    name: "swin_v2_base"
    pretrained: true
    freeze_encoder: true
    unfreeze_after_epoch: 15
  decoder_dropout: 0.2
```

## Expected Benefits

1. **Better feature extraction**: Pretrained on 302M remote sensing labels
2. **Faster convergence**: Transfer learning reduces training time
3. **Higher accuracy**: Expected 1.7-2.3% IoU improvement (based on ISPRS studies)
4. **Domain-specific**: Trained on satellite/aerial imagery, not ImageNet

## Risks & Mitigations

### Risk 1: 5-channel input mismatch
- **Mitigation**: Input adapter layer (see Step 6)

### Risk 2: Encoder feature dimension mismatch
- **Mitigation**: Dynamic decoder dimension calculation based on encoder

### Risk 3: Dependency conflicts
- **Mitigation**: Test in isolated environment, pin versions in `pyproject.toml`

### Risk 4: Memory issues (larger models)
- **Mitigation**:
  - Use ResNet50 instead of ResNet152 initially
  - Reduce batch size if needed
  - Use gradient checkpointing

## Implementation Checklist

- [ ] Step 1: Install dependencies (`satlaspretrain-models`, `torchgeo`)
- [ ] Step 2: Create model factory (`src/models/factory.py`)
- [ ] Step 3: Implement SatlasPretrain U-Net (`src/models/satlaspretrain_unet.py`)
- [ ] Step 4: Update training script (`scripts/train_model.py`)
- [ ] Step 5: Update configuration (`configs/training_config.yaml`)
- [ ] Step 6: Implement 5-channel input adapter
- [ ] Step 7: Add encoder unfreezing logic
- [ ] Step 8: Update MLflow logging for encoder info
- [ ] Step 9: Write unit tests
- [ ] Step 10: Test backward compatibility (baseline UNet)
- [ ] Step 11: Run integration test (1-2 epochs)
- [ ] Step 12: Update documentation

## Future Extensions

This architecture allows easy addition of:
- **Option 3**: Segmentation Models PyTorch (ResNet34/50 with ImageNet pretraining)
- **Option 4**: Custom encoder architectures
- **Option 5**: Ensemble models

Simply add new architecture class and register in factory.

## References

- SatlasPretrain: https://blog.allenai.org/satlaspretrain-models-foundation-models-for-satellite-and-aerial-imagery
- SatlasPretrain GitHub: https://github.com/allenai/satlaspretrain_models
- TorchGeo: https://github.com/microsoft/torchgeo
- Research: "Semantic Segmentation of High-Resolution Remote Sensing Images with Improved U-Net Based on Transfer Learning"
