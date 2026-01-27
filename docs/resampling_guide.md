# Resampling Guide for DEM and Slope Layers

## Overview

When combining raster layers with different resolutions (e.g., RGB imagery at 0.2m and DEM at 2.0m), we need to resample the lower-resolution layers to match the higher-resolution reference before stacking them.

## Recommended Resampling Methods

### For DEM (Elevation) Data

**Bilinear (Recommended - Default)**
- **Why**: Smooth interpolation that preserves terrain characteristics
- **Pros**: 
  - Smooth continuous surface
  - No overshooting/undershooting (doesn't create artificial peaks/valleys)
  - Preserves general terrain shape
  - Computationally efficient
  - Standard practice in GIS and remote sensing
- **Cons**: 
  - Slightly less smooth than cubic methods
- **Best for**: Most terrain types, general use

**Cubic (Alternative)**
- **Why**: Even smoother interpolation
- **Pros**: 
  - Very smooth surface
  - Better for very smooth terrain
- **Cons**: 
  - Can overshoot/undershoot values (create peaks/valleys that don't exist)
  - May introduce artifacts in steep terrain
  - More computationally expensive
- **Best for**: Very smooth terrain, when you need maximum smoothness

**Nearest Neighbor (NOT Recommended)**
- **Why**: Preserves exact values
- **Cons**: 
  - Creates blocky, stepped appearance
  - Not suitable for continuous elevation data
  - Will create artifacts in the final model
- **Best for**: Categorical data only (not elevation)

### For Slope Data

Since slope is derived from DEM, use the **same method as DEM**:
- **Bilinear (Recommended)**: Same reasons as DEM
- **Cubic (Alternative)**: If using cubic for DEM, use cubic for slope too

## Current Implementation

The resampling script (`scripts/resample_raster.py`) defaults to **bilinear**, which is the recommended choice for DEM and slope data.

### Usage

```bash
# Default (bilinear - recommended)
poetry run python scripts/resample_raster.py \
    -i data/raw/raster/dem/dem_from_arcticDEM_cropped2.tif \
    -r data/raw/raster/imagery/qaanaaq_rgb_0_2m.tif \
    -o data/processed/raster/dem_resampled.tif

# Explicitly specify bilinear
poetry run python scripts/resample_raster.py \
    -i data/raw/raster/dem/dem_from_arcticDEM_cropped2.tif \
    -r data/raw/raster/imagery/qaanaaq_rgb_0_2m.tif \
    -o data/processed/raster/dem_resampled.tif \
    --method bilinear

# Use cubic for smoother results (if needed)
poetry run python scripts/resample_raster.py \
    -i data/raw/raster/dem/dem_from_arcticDEM_cropped2.tif \
    -r data/raw/raster/imagery/qaanaaq_rgb_0_2m.tif \
    -o data/processed/raster/dem_resampled.tif \
    --method cubic
```

## Why Bilinear for Elevation?

1. **Terrain Continuity**: Elevation changes smoothly in nature. Bilinear interpolation respects this continuity.

2. **No Artifacts**: Unlike cubic methods, bilinear doesn't overshoot values, preventing artificial peaks or valleys.

3. **Standard Practice**: Bilinear is the standard resampling method for elevation data in GIS software (QGIS, ArcGIS, etc.).

4. **CNN Training**: For CNN models, smooth continuous features are better than blocky or oversmoothed ones. Bilinear provides the right balance.

5. **Computational Efficiency**: Faster than cubic methods, important when processing large datasets.

## When to Use Cubic?

Consider cubic if:
- Your terrain is very smooth (glacial, coastal plains)
- You need maximum smoothness
- You're willing to accept potential overshooting artifacts
- Processing time is not a concern

## Summary

**For this project**: **Bilinear (default)** is the recommended and correct choice for DEM and slope resampling. It provides smooth, artifact-free elevation data suitable for CNN training.
