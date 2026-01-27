# Per-Tile Baseline Analysis

## Overview

This approach calculates baseline metrics for **each individual tile** before training, allowing you to:
- **Distinguish real improvements** from tile-specific characteristics
- **Identify which tiles** the model is genuinely learning vs just matching baseline
- **Understand tile difficulty** based on class imbalance and baseline performance

## Why This Matters

With only **2.1% lobe pixels** overall, a model could achieve low MAE by:
- Predicting 0 everywhere (baseline MAE ≈ 0.45)
- Some tiles might have higher/lower lobe ratios, making them easier/harder

**Per-tile baselines** help you understand:
- Is the model improving on **this specific tile**, or just matching its baseline?
- Are improvements consistent across tiles, or only on "easy" tiles?
- Which tiles need more attention during training?

## What Gets Computed

For each tile, we calculate:

### 1. Class Imbalance
- Lobe pixels (value >= threshold)
- Background pixels (value < threshold)
- Lobe fraction (percentage)

### 2. Baseline MAE Strategies
- **Predict 0**: MAE if we predict background everywhere
- **Predict mean**: MAE if we predict the tile's mean value
- **Predict median**: MAE if we predict the tile's median value
- **Weighted optimal**: Best possible naive strategy (0 for background, mean for lobes)

### 3. Baseline RMSE
- Predict 0 and predict mean strategies

### 4. Baseline IoU
- IoU if we predict 0 everywhere (should be 0.0)

### 5. Per-Class Baselines
- MAE for lobe pixels separately
- MAE for background pixels separately

## Usage

### Step 1: Re-filter Tiles with Baselines

The baseline computation is now **automatic** when filtering tiles:

```bash
poetry run python scripts/filter_tiles.py \
    --features data/processed/tiles/train/features \
    --targets data/processed/tiles/train/targets \
    --output data/processed/tiles/train/filtered_tiles.json \
    --exclude-background \
    --lobe-threshold 5.0
```

This will add `baseline_metrics` to each tile in the JSON file.

### Step 2: View Baseline Statistics

Check the filtered_tiles.json - each tile now has:

```json
{
  "tile_id": "tile_0000",
  "target_stats": {
    "baseline_metrics": {
      "class_imbalance": {
        "lobe_fraction": 0.041,
        "lobe_pixels": 2696,
        "lobe_threshold": 5.0
      },
      "baseline_mae": {
        "predict_zero": 0.373,
        "weighted_optimal": 0.135
      },
      ...
    }
  }
}
```

### Step 3: Compare During Training

During validation, you can compare model performance to per-tile baselines:

- **Model MAE < Baseline MAE (predict_zero)**: Model is improving!
- **Model MAE < Baseline MAE (weighted_optimal)**: Model is doing better than best naive strategy!
- **Model MAE ≈ Baseline MAE**: Model might just be matching baseline

### Step 4: Post-Training Analysis

After training, use the analysis script to compare predictions:

```bash
poetry run python scripts/analyze_per_tile_performance.py \
    --filtered-tiles data/processed/tiles/train/filtered_tiles.json \
    --targets-dir data/processed/tiles/train/targets \
    --predictions path/to/model/predictions \
    --iou-threshold 5.0 \
    --output analysis_results.json
```

## Example Interpretation

### Tile A (High Lobe Fraction: 8%)
- Baseline MAE (predict 0): 0.65
- Model MAE: 0.40
- **Interpretation**: Model is genuinely improving (0.25 better than baseline)

### Tile B (Low Lobe Fraction: 0.5%)
- Baseline MAE (predict 0): 0.15
- Model MAE: 0.14
- **Interpretation**: Model is barely better than baseline - might just be matching tile characteristics

### Tile C (Medium Lobe Fraction: 2%)
- Baseline MAE (predict 0): 0.45
- Baseline MAE (weighted optimal): 0.28
- Model MAE: 0.25
- **Interpretation**: Model is better than naive baseline AND optimal strategy - real learning!

## Benefits

1. **Honest Evaluation**: Know if improvements are real or just tile-specific
2. **Identify Problem Tiles**: Find tiles where model struggles despite low baseline
3. **Track Progress**: See which tiles improve over epochs
4. **Data Quality**: Identify tiles with unusual characteristics

## Integration with Training

The baseline metrics are stored in `filtered_tiles.json` and can be:
- Loaded during training to track per-tile progress
- Used to weight tiles differently (focus on harder tiles)
- Analyzed post-training to understand model behavior

## Next Steps

Consider adding:
- **Per-tile metrics logging** during validation (track which tiles improve)
- **Tile difficulty ranking** based on baseline vs actual performance
- **Visualization** of per-tile improvements over epochs
