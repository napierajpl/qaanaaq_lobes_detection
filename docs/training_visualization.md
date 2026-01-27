# Training Visualization

## Overview

Training metrics are automatically visualized and logged to MLflow during training. This provides clear insights into model performance and progress.

## Charts Generated

### 1. MAE Comparison (`mae_comparison.png`)
- **Model MAE** (blue line): Validation MAE across epochs
- **Baseline MAE** (red dashed line): Constant baseline MAE (predict 0 everywhere)
- **Purpose**: See if model is improving beyond baseline

### 2. Loss Plot (`loss.png`)
- **Train Loss** (blue line): Training loss across epochs
- **Val Loss** (red line): Validation loss across epochs
- **Purpose**: Monitor overfitting and training progress

### 3. IoU Plot (`iou.png`)
- **Val IoU** (green line): Validation Intersection over Union across epochs
- **Purpose**: Track segmentation quality (higher is better)

### 4. Improvement Percentage (`improvement_percent.png`)
- **Bar chart**: Percent improvement over baseline per epoch
- **Green bars**: Better than baseline (positive %)
- **Red bars**: Worse than baseline (negative %)
- **Purpose**: Quick visual indicator of improvement

## Where to Find Charts

Charts are automatically logged to MLflow:
- **Location**: `mlruns/<experiment>/<run_id>/artifacts/plots/`
- **View in MLflow UI**: Open MLflow UI and navigate to the run's "Artifacts" tab

## Percent Improvement Calculation

The percent improvement is calculated as:
```
improvement_percent = (improvement / baseline_mae) * 100
```

Where:
- `improvement = baseline_mae - model_mae`
- Positive % = model is better than baseline
- Negative % = model is worse than baseline

## Example Output

During training, you'll see:
```
Epoch 1 [Val]: loss=0.4523, mae=0.3821, iou=0.0000
  Baseline MAE: 0.3727 | Model MAE: 0.3821 | [WORSE] (delta: -0.0094, -2.52%)
```

This shows:
- **Delta**: Absolute difference (baseline - model)
- **Percent**: Relative improvement (-2.52% means model is 2.52% worse)

## Implementation

Visualization code is in `src/training/visualization.py` to keep the project organized. Charts are created after training completes and logged to MLflow automatically.
