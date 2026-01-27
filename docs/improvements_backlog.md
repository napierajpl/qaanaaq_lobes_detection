# Improvements Backlog

This file tracks potential improvements, ideas, and future work for the lobe detection project. Items are organized by priority and category.

## High Priority

### Loss Functions

1. **Focal Loss** ⭐ (Highest Priority)
   - **Why**: Designed specifically for extreme class imbalance (93.5% background vs 6.5% lobes)
   - **What**: Down-weights easy background examples, focuses learning on hard/lobe pixels
   - **Status**: Not implemented
   - **Reference**: Identified in daily diary 2026-01-21

2. **Increase Encouragement Weight**
   - **Why**: Current weight=10.0 insufficient to break local minimum
   - **What**: Try 50.0 or 100.0 instead of 10.0 for stronger penalty
   - **Status**: Not tested
   - **Reference**: Identified in daily diary 2026-01-21

### Architecture

3. **Pretrained Encoder** ⭐ (High Impact Expected) ✅ **IMPLEMENTED**
   - **Why**: Better feature extraction from start, transfer learning benefits, proven to improve remote sensing segmentation
   - **What**: Replace U-Net encoder with pretrained backbone:
     - **Option A (Easiest)**: Use `segmentation-models-pytorch` with ResNet34/ResNet50 encoder (ImageNet pretrained) - Not implemented
     - **Option B (Best for RS)**: Use SatlasPretrain ResNet50 (trained on 302M remote sensing labels) ✅ **IMPLEMENTED**
     - **Option C**: Use TorchGeo pretrained models for multispectral imagery - Not implemented
   - **Implementation Notes**:
     - Adapt first conv layer for 5 channels (RGB+DEM+Slope) instead of 3 ✅
     - Freeze encoder initially, then fine-tune after convergence ✅
     - Expected improvement: 1.7-2.3% IoU gain (based on ISPRS studies)
   - **Status**: ✅ Implemented - Ready for testing (requires satlaspretrain-models installation)
   - **Implementation Summary**: See `docs/implementation_plans/satlaspretrain_implementation_summary.md`
   - **Implementation Plan**: See `docs/implementation_plans/satlaspretrain_integration.md`
   - **References**: 
     - Identified in daily diary 2026-01-21
     - Research: "Semantic Segmentation of High-Resolution Remote Sensing Images with Improved U-Net Based on Transfer Learning"
     - SatlasPretrain: https://blog.allenai.org/satlaspretrain-models-foundation-models-for-satellite-and-aerial-imagery
     - Segmentation Models PyTorch: https://github.com/qubvel/segmentation_models.pytorch

## Medium Priority

### Training Strategy

4. **Warm-up Training**
   - **Why**: Escape local minimum before fine-tuning
   - **What**: High LR (0.1) for 5-10 epochs, then reduce to normal LR
   - **Status**: Not implemented
   - **Reference**: Identified in daily diary 2026-01-21

5. **Learning Rate Adjustments**
   - **Why**: LR=0.01 may be too high, causing early plateau
   - **What**: Experiment with lower initial LR (0.001-0.005) or better scheduling
   - **Status**: Partially addressed (scheduler exists but doesn't trigger)

### Data

6. **Data Augmentation**
   - **Why**: Increase effective dataset size, improve generalization
   - **What**: 
     - Geometric: horizontal/vertical flips, rotations (90°, 180°, 270°)
     - Photometric (RGB only): brightness, contrast adjustments
     - Note: Don't augment DEM/Slope (preserve physical meaning)
   - **Status**: Not implemented
   - **Reference**: Discussed in conversation 2026-01-22

7. **Class-Balanced Sampling**
   - **Why**: Most batches are mostly background, model sees few lobe examples
   - **What**: Oversample tiles with high lobe density during training
   - **Status**: Not implemented

### Tile Size

8. **Experiment with Larger Tiles**
   - **Why**: More spatial context around lobes, better for larger features
   - **What**: Try 384×384 or 512×512 tiles (requires retiling, reduces batch size)
   - **Status**: Not tested
   - **Note**: Lower priority - current issue is learning, not context

## Low Priority / Future Work

9. **Strip Convolutions for Linear Features**
   - **Why**: Lobes are linear/elongated features, strip convolutions excel at high aspect ratio objects
   - **What**: Incorporate strip convolution concepts from Strip R-CNN (82.75% mAP on DOTA-v1.0)
   - **Complexity**: High - requires architecture modifications
   - **Reference**: Strip R-CNN paper - "Large Strip Convolution for Remote Sensing Object Detection"
   - **Status**: Research phase

10. **Two-Stage Training**
   - **Why**: Separate detection from regression
   - **What**: 
     - Stage 1: Binary segmentation (lobe vs background) with Dice/Focal Loss
     - Stage 2: Fine-tune for regression (proximity values) on lobe pixels only
   - **Status**: Not implemented
   - **Complexity**: High - requires pipeline changes

11. **Attention Mechanisms**
    - **Why**: Focus model attention on important regions
    - **What**: Add attention layers to U-Net architecture
    - **Status**: Not implemented

12. **Deep Supervision**
    - **Why**: Add auxiliary losses at intermediate layers
    - **What**: Compute loss at multiple decoder levels
    - **Status**: Not implemented

13. **Ensemble Methods**
    - **Why**: Combine multiple models for better predictions
    - **What**: Train multiple models, average predictions
    - **Status**: Not implemented

## Completed / In Progress

- ✅ Dropout added to decoder (0.2)
- ✅ Gradient clipping implemented
- ✅ Learning rate scheduling infrastructure
- ✅ Early stopping
- ✅ Per-tile baseline comparison
- ✅ Training visualization module
- ✅ MLflow integration
- ✅ SatlasPretrain U-Net architecture with pretrained encoders (Jan 23, 2026)
  - Model factory system for architecture switching
  - 5-channel input adapter
  - Encoder freezing/unfreezing support
  - Support for ResNet50, ResNet152, Swin-v2-Base, Swin-v2-Tiny

## Notes

- Items are added as they are identified during experiments
- Priority may change based on results
- When an item is implemented, move it to "Completed" and reference the daily diary entry
- Reference daily diary entries to understand context and rationale
