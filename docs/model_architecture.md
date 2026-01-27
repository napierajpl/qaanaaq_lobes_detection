# Model Architecture and Training Guide

## Overview

This document explains how our lobe detection model works, from input data to predictions. The model uses a U-Net architecture to predict proximity maps that indicate the distance and location of glacial lobes in satellite imagery.

## Model Architecture: U-Net

### Architecture Overview

The model uses a **Classic U-Net** architecture, which is a convolutional neural network designed for semantic segmentation tasks. The U-shaped structure consists of:

1. **Encoder (Contracting Path)**: Extracts features by downsampling
2. **Bottleneck**: Deepest layer with most abstract features
3. **Decoder (Expanding Path)**: Reconstructs spatial resolution by upsampling
4. **Skip Connections**: Preserve fine-grained details from encoder to decoder

### Architecture Details

**Input**: 5-channel tensor (256×256 pixels)
- Channel 0-2: RGB imagery (normalized to [0, 1])
- Channel 3: DEM (Digital Elevation Model, standardized)
- Channel 4: Slope (standardized)

**Output**: 1-channel tensor (256×256 pixels)
- Continuous values from 0 to 10 representing proximity to lobes

**Network Structure**:

```
Input (5×256×256)
  ↓
Encoder:
  - enc1: 5 → 64 channels (256×256)
  - enc2: 64 → 128 channels (128×128)
  - enc3: 128 → 256 channels (64×64)
  - enc4: 256 → 512 channels (32×32)
  ↓
Bottleneck: 512 → 1024 channels (16×16)
  ↓
Decoder (with skip connections):
  - dec4: 1024 → 512 channels (32×32) + skip(enc4)
  - dec3: 512 → 256 channels (64×64) + skip(enc3)
  - dec2: 256 → 128 channels (128×128) + skip(enc2)
  - dec1: 128 → 64 channels (256×256) + skip(enc1)
  ↓
Output: 64 → 1 channel (256×256)
```

**Key Components**:

- **DoubleConv Blocks**: Each encoder/decoder layer uses two 3×3 convolutions with BatchNorm and ReLU
- **MaxPooling**: Reduces spatial dimensions in encoder (2×2, stride 2)
- **Transposed Convolutions**: Upsamples in decoder (2×2, stride 2)
- **Skip Connections**: Concatenates encoder features with decoder features at same resolution
- **Dropout**: Applied in decoder only (0.2) to prevent overfitting
- **Final Layer**: 1×1 convolution maps features to single-channel output

**Total Parameters**: ~31 million

## Input Data Format

### Feature Tiles (5 Channels)

The model receives 256×256 pixel tiles with 5 channels:

1. **RGB Bands (Channels 0-2)**
   - Source: Satellite imagery
   - Normalization: `pixel_value / 255.0` (range [0, 1])
   - Purpose: Visual features (texture, color, patterns)

2. **DEM Band (Channel 3)**
   - Source: Digital Elevation Model from ArcticDEM
   - Normalization: Standardized using mean and std from training data
   - Purpose: Elevation information (lobes may have characteristic elevation patterns)

3. **Slope Band (Channel 4)**
   - Source: Derived from DEM
   - Normalization: Standardized using mean and std from training data
   - Purpose: Terrain slope (lobes may have characteristic slopes)

### Target Tiles (Proximity Maps)

The model learns to predict **proximity maps** with values 0-20:

- **0**: Background (far from lobes, >20 pixels away)
- **1-19**: Proximity values (closer to lobes, 1-19 pixels away)
- **20**: Lobe center (actual lobe pixels)

**How Proximity Maps are Generated**:

1. Start with binary raster (lobes = 1, background = 0)
2. Calculate Euclidean distance transform from lobe pixels
3. Apply decay function: `value = max(0, 20 - distance)`
4. Pixels beyond 20 pixels from lobes get value 0

**Note**: Proximity zone expanded from 10px to 20px (January 22, 2026) to improve class balance and make threshold crossing easier.

**Why Proximity Maps?**

- Provides spatial context (not just binary classification)
- Helps model learn gradual transitions
- Encodes distance information useful for detection
- Allows regression instead of classification (smoother gradients)

## Training Process

### Data Preparation

1. **Tiling**: Large rasters are divided into 256×256 overlapping tiles (30% overlap)
2. **Filtering**: Tiles are filtered to remove:
   - Empty RGB tiles (no valid imagery)
   - Background-only tiles (no lobe pixels)
3. **Normalization**: Statistics computed from training set for DEM/slope standardization

### Training Loop

For each epoch:

1. **Training Phase**:
   - Model processes batches of tiles
   - Loss is computed and backpropagated
   - Weights are updated via Adam optimizer
   - Gradient clipping prevents exploding gradients

2. **Validation Phase**:
   - Model evaluates on held-out validation set
   - Metrics computed: loss, MAE, RMSE, IoU
   - Best model saved based on validation loss

3. **Learning Rate Scheduling**:
   - `ReduceLROnPlateau`: Reduces LR when validation loss plateaus
   - Patience: 10 epochs
   - Factor: 0.5 (halves learning rate)
   - Min LR: 0.0001

4. **Early Stopping**:
   - Stops training if no improvement for 20 epochs
   - Prevents overfitting and saves time

### Loss Functions

The model can use several loss functions (configured in `training_config.yaml`):

#### 1. Smooth L1 Loss (Huber Loss)
- Basic regression loss
- Smooth transition between L1 and L2
- Good for general regression tasks

#### 2. Weighted Smooth L1 Loss
- Applies higher weight to lobe pixels (value ≥ threshold)
- Handles class imbalance (93.5% background vs 6.5% lobes)
- Weight: 5.0 for lobe pixels

#### 3. Dice Loss
- Directly optimizes Dice coefficient (similar to IoU)
- Good for segmentation tasks
- Binarizes predictions/targets at threshold

#### 4. IoU Loss (Jaccard Loss)
- Directly optimizes Intersection over Union
- 1 - IoU as loss
- Binarizes predictions/targets at threshold

#### 5. Soft IoU Loss
- Uses sigmoid instead of hard thresholding
- Provides smooth gradients
- Better for training than hard IoU

#### 6. Encouragement Loss ⭐ (Recommended)
- **MSE Component**: Standard regression loss for lobe pixels
- **Penalty Component**: Penalizes predictions < threshold when targets ≥ threshold
- **Formula**: `loss = MSE + encouragement_weight × penalty`
- **Why it works**: Directly pushes predictions above threshold

#### 7. Combined Loss
- Combines IoU loss + Weighted Smooth L1
- Balances segmentation and regression objectives
- Can use soft IoU for better gradients

**Current Recommendation**: Use `"encouragement"` loss with `encouragement_weight: 10.0` to directly optimize for predictions above threshold.

## Evaluation Metrics

### Mean Absolute Error (MAE)
- Average absolute difference between predictions and targets
- Lower is better
- Formula: `MAE = mean(|pred - target|)`

### Root Mean Squared Error (RMSE)
- Square root of average squared differences
- Penalizes large errors more than MAE
- Formula: `RMSE = sqrt(mean((pred - target)²))`

### Intersection over Union (IoU)
- Measures overlap between predicted and target lobe areas
- Binarizes at threshold (default: 1.0)
- Formula: `IoU = intersection / union`
- Range: [0, 1], higher is better
- **IoU = 0** means no overlap (model predicts all values < threshold)

### Baseline Comparison
- Compares model MAE to naive baseline (predicting 0 everywhere)
- Baseline MAE: ~0.329
- Model should beat baseline to be useful

## How Predictions Work

### Inference Process

1. **Load Input Tile**: 5-channel feature tile (256×256)
2. **Normalize**: Apply same normalization as training
3. **Forward Pass**: Model processes through U-Net
4. **Output**: Single-channel proximity map (256×256)
5. **Post-processing** (optional):
   - Threshold at 1.0 to create binary mask
   - Or use continuous values for distance information

### Interpreting Predictions

- **Values < 1.0**: Background (no lobes nearby)
- **Values ≥ 1.0**: Lobe area (includes proximity zones)
- **Values ≥ 5.0**: Strong lobe signal
- **Values = 10.0**: Lobe center

### Current Model Behavior

⚠️ **Known Issue**: Current model predicts values ~0.4 for all pixels, never crossing the 1.0 threshold. This results in IoU = 0.0.

**Why this happens**:
- Class imbalance (93.5% background) dominates loss
- Model minimizes loss by predicting low values everywhere
- Current loss functions don't provide strong enough gradients

**Solution**: Use `"encouragement"` loss with high `encouragement_weight` to directly penalize predictions below threshold.

## Data Flow

```
Raw Data
  ↓
1. Rasterize vector layers → Binary raster (lobes = 1, background = 0)
  ↓
2. Generate proximity map → Proximity values (0-10)
  ↓
3. Resample DEM/Slope → Match RGB resolution
  ↓
4. Create VRT stack → Combine RGB + DEM + Slope (5 bands)
  ↓
5. Create tiles → 256×256 overlapping tiles
  ↓
6. Filter tiles → Remove empty/background-only tiles
  ↓
7. Normalize → Compute statistics, normalize features
  ↓
8. Training → Model learns to predict proximity maps
  ↓
9. Inference → Model predicts proximity values for new tiles
```

## Model Configuration

Key parameters in `configs/training_config.yaml`:

- **Architecture**: `unet`
- **Input channels**: 5 (RGB + DEM + Slope)
- **Output channels**: 1 (proximity map)
- **Base channels**: 64
- **Dropout**: 0.2 (decoder only)
- **Batch size**: 16-32
- **Learning rate**: 0.01 (with scheduling)
- **Loss function**: `encouragement` (recommended)
- **IoU threshold**: 1.0 (pixels ≥ 1.0 considered lobes)

## Future Improvements

1. **Pretrained Encoder**: Use ResNet34 encoder for better feature extraction
2. **Attention Mechanisms**: Add attention to focus on important regions
3. **Deep Supervision**: Add auxiliary losses at intermediate layers
4. **Data Augmentation**: Rotations, flips, color jitter
5. **Ensemble Methods**: Combine multiple models for better predictions

## References

- Original U-Net paper: Ronneberger et al., 2015
- Proximity maps: Distance transform-based feature encoding
- This implementation: Classic U-Net without pretrained encoder (baseline model)
