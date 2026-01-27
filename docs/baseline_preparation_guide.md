# Baseline Preparation Guide

## Should You Prepare Baselines Before Full Training?

**YES - Always prepare baselines before training on full dataset.**

## Why It's Important

### 1. **Evaluation Context**
- Baselines tell you if improvements are **real** or just matching tile characteristics
- Without baselines, you can't tell if MAE=0.5 is good or bad for a specific tile
- Example: Tile with 0.1% lobes has baseline MAE=0.05 → MAE=0.5 is terrible!

### 2. **One-Time Cost**
- Baseline computation happens **once** during filtering
- Takes ~1-2 seconds per tile (just reading and computing stats)
- For 10,000 tiles: ~3-6 hours (one-time)
- **Worth it** for the evaluation insights you get

### 3. **Already Integrated**
- Baselines are computed **automatically** when filtering tiles
- No extra step needed - just run the normal pipeline
- Stored in `filtered_tiles.json` for later use

### 4. **Training Integration**
- Training script uses baselines for:
  - Real-time comparison during validation
  - Percent improvement calculation
  - Better understanding of model progress

## How It Works

### Automatic Baseline Computation

When you run the filtering step, baselines are computed automatically:

```bash
poetry run python scripts/filter_tiles.py \
    --features data/processed/tiles/train/features \
    --targets data/processed/tiles/train/targets \
    --output data/processed/tiles/train/filtered_tiles.json \
    --exclude-background \
    --lobe-threshold 5.0
```

**Baselines are computed by default** (unless you use `--no-baselines`).

### What Gets Computed Per Tile

For each tile:
- **Class imbalance**: Lobe vs background pixel counts
- **Baseline MAE**: Predict 0, predict mean, predict median, weighted optimal
- **Baseline RMSE**: Predict 0, predict mean
- **Baseline IoU**: Predict 0 everywhere
- **Per-class baselines**: Separate MAE for lobe vs background pixels

### Storage

All baseline metrics are stored in `filtered_tiles.json`:

```json
{
  "tiles": [
    {
      "tile_id": "tile_0000",
      "target_stats": {
        "baseline_metrics": {
          "baseline_mae": {
            "predict_zero": 0.373,
            "weighted_optimal": 0.135
          },
          ...
        }
      }
    }
  ]
}
```

## Performance Considerations

### Time Cost

- **Per tile**: ~1-2 seconds (read tile + compute stats)
- **10,000 tiles**: ~3-6 hours
- **50,000 tiles**: ~14-28 hours

### Memory Cost

- Minimal - processes one tile at a time
- No significant memory overhead

### When to Skip

Only skip baselines if:
- ❌ You're just testing the pipeline (use `--no-baselines`)
- ❌ You have millions of tiles and time is critical
- ✅ **Otherwise: Always compute baselines!**

## Production Pipeline

The `prepare_training_data.py` script automatically:
1. Creates tiles
2. Filters tiles (with baselines computed automatically)
3. Ready for training

**No extra step needed** - baselines are part of the normal pipeline.

## Recommendation

**For full dataset training:**
1. ✅ Run normal `prepare_training_data.py` (production mode)
2. ✅ Baselines will be computed automatically during filtering
3. ✅ Wait for it to complete (one-time cost)
4. ✅ Then start training with baseline-aware evaluation

**The time investment is worth it** for the evaluation insights you'll get throughout training and after.
