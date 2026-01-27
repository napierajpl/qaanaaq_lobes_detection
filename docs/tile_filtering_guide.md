# Tile Filtering Guide

## Overview

The tile filtering step removes low-quality tiles from the training dataset to improve model training efficiency and quality.

## Filtering Criteria

### 1. Empty RGB Tiles (Always Excluded)

**Problem**: Some tiles contain no RGB imagery data (empty/missing coverage areas).

**Solution**: Tiles with less than 1% valid RGB pixels are automatically excluded.

**Rationale**:
- Training on empty tiles provides no useful information
- Wastes computational resources
- Can confuse the model

### 2. Background-Only Tiles (Configurable)

**Problem**: Many tiles contain only background (no lobe targets).

**Considerations**:
- **Include them**: Helps model learn what background looks like, prevents overfitting to positive examples
- **Exclude them**: Focuses training on tiles with actual targets, may improve precision

**Default Behavior**: Currently set to **exclude** background-only tiles (`--exclude-background` flag).

**Recommendation**:
- **Start with exclusion** for initial training (faster convergence, better focus on positive examples)
- **Experiment with inclusion** if model struggles with false positives (needs better background understanding)

## Usage

### Basic Filtering (Exclude Empty RGB + Background-Only)

```bash
poetry run python scripts/filter_tiles.py \
    --features data/processed/tiles/dev/train/features \
    --targets data/processed/tiles/dev/train/targets \
    --output data/processed/tiles/dev/train/filtered_tiles.json \
    --exclude-background
```

### Include Background-Only Tiles

```bash
poetry run python scripts/filter_tiles.py \
    --features data/processed/tiles/dev/train/features \
    --targets data/processed/tiles/dev/train/targets \
    --output data/processed/tiles/dev/train/filtered_tiles.json
    # Note: --exclude-background flag NOT used
```

### Filter with Minimum Target Coverage

Only include tiles with at least X% positive pixels:

```bash
poetry run python scripts/filter_tiles.py \
    --features data/processed/tiles/dev/train/features \
    --targets data/processed/tiles/dev/train/targets \
    --output data/processed/tiles/dev/train/filtered_tiles.json \
    --min-target-coverage 0.01  # At least 1% of pixels must be positive
```

## Output Format

The filtered tile list is saved as JSON:

```json
{
  "filter_config": {
    "min_rgb_coverage": 0.01,
    "include_background_only": false,
    "min_target_coverage": null
  },
  "stats": {
    "total_tiles": 36,
    "rgb_invalid": 2,
    "background_only": 5,
    "valid_tiles": 29
  },
  "tiles": [
    {
      "tile_id": "tile_0000",
      "features_path": "features_combined_cropped1024x1024/tile_0000.tif",
      "targets_path": "rasterized_lobes_raw_by_code_cropped1024x1024_proximity10px/tile_0000.tif",
      "rgb_valid": true,
      "has_targets": true,
      "rgb_stats": {
        "valid_pixels": 65536,
        "total_pixels": 65536,
        "coverage_ratio": 1.0
      },
      "target_stats": {
        "positive_pixels": 1234,
        "total_pixels": 65536,
        "coverage_ratio": 0.0188,
        "max_value": 10.0
      }
    }
  ]
}
```

## Integration with Training Pipeline

The filtering step is automatically included in:
- `scripts/prepare_training_data_dev.sh` (Step 7)
- `scripts/prepare_training_data.sh` (Step 5)

## Using Filtered Tiles in Training

Training scripts should load the filtered tile list:

```python
import json
from pathlib import Path

# Load filtered tiles
with open("data/processed/tiles/dev/train/filtered_tiles.json") as f:
    tile_data = json.load(f)

# Get list of valid tile pairs
valid_tiles = tile_data["tiles"]

# Access tile paths
for tile_info in valid_tiles:
    features_path = Path("data/processed/tiles/dev/train/features") / tile_info["features_path"]
    targets_path = Path("data/processed/tiles/dev/train/targets") / tile_info["targets_path"]
    # Load and use for training...
```

## Statistics Interpretation

After filtering, you'll see a summary:

```
=== Tile Filtering Summary ===
Total tiles processed: 36
  - RGB invalid (empty): 2 (5.6%)
  - Background only (excluded): 5 (13.9%)
  - Valid tiles: 29 (80.6%)
```

**What to expect**:
- **RGB invalid**: Usually 0-10% depending on imagery coverage
- **Background only**: Can be 20-50% or more depending on lobe density
- **Valid tiles**: Remaining tiles suitable for training

## Recommendations

1. **Initial Training**: Use `--exclude-background` to focus on positive examples
2. **If Model Overfits**: Try including background-only tiles (remove `--exclude-background`)
3. **If Too Many False Positives**: Include background-only tiles to improve background recognition
4. **If Dataset Too Small**: Include background-only tiles to increase training data
5. **Monitor Statistics**: Check filtering stats to understand your dataset composition

## Advanced: Custom Filtering Logic

You can modify `src/data_processing/tile_filter.py` to add custom filtering criteria:

- Minimum/maximum target coverage ratios
- Spatial distribution requirements
- Quality metrics (e.g., variance, edge density)
- Custom validation functions
