# Qaanaaq Lobes Detection

CNN pipeline for detecting lobes using raster and vector geospatial data.

## Overview

This project implements a deep learning pipeline for detecting glacial lobes from high-resolution aerial imagery (0.2m/pixel) using a U-Net architecture. The system processes multi-channel geospatial data (RGB + DEM + Slope) and outputs proximity maps indicating lobe locations.

## Features

- **Multiple Architecture Support**: Baseline UNet and SatlasPretrain U-Net with pretrained encoders
- **Experiment Tracking**: MLflow integration for tracking experiments, metrics, and artifacts
- **Data Management**: Tile-based processing with filtering and quality control
- **Geospatial Visualization**: QGIS-compatible shapefiles for tile visualization
- **Comprehensive Metrics**: MAE, RMSE, IoU, and baseline comparisons

## Setup

### Using Poetry (Recommended)

1. Install Poetry if not already installed:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

### Manual Installation

If you prefer not to use Poetry, install dependencies manually:

```bash
pip install geopandas rasterio numpy pyyaml
```

## Project Structure

See `docs/PROJECT_STRUCTURE.md` for detailed folder organization.

## Usage

### Training a Model

1. Prepare training data:
   ```bash
   poetry run python scripts/prepare_training_data.py
   ```

2. Train model:
   ```bash
   poetry run python scripts/train_model.py
   ```

3. View results in MLflow:
   ```bash
   poetry run python scripts/start_mlflow_ui.py
   # Navigate to http://127.0.0.1:5001
   ```

### Data Preparation

- **Rasterize Vector Layer**:
  ```bash
  poetry run python scripts/rasterize_vector.py
  ```

- **Generate Proximity Maps**:
  ```bash
  poetry run python scripts/generate_proximity_map.py
  ```

- **Create Tiles**:
  ```bash
  poetry run python scripts/create_tiles.py
  ```

### Visualization

- **Generate Tile Index Shapefile** (for QGIS):
  ```bash
  poetry run python scripts/generate_tile_index_shapefile.py
  ```

## Architecture Options

The project supports multiple model architectures:

1. **Baseline UNet**: Standard U-Net without pretrained encoder
2. **SatlasPretrain U-Net**: U-Net with pretrained SatlasPretrain encoder (ResNet50, ResNet152, Swin-v2-Base, Swin-v2-Tiny)

Configure in `configs/training_config.yaml`:
```yaml
model:
  architecture: "satlaspretrain_unet"  # or "unet"
  encoder:
    name: "swin_v2_base"  # Recommended for aerial imagery
    pretrained: true
    freeze_encoder: true
    unfreeze_after_epoch: 10
```

## Dependencies

- **Required**: See `pyproject.toml` for full list
- **Optional**: `satlaspretrain-models` (install manually: `poetry run pip install satlaspretrain-models`)

## Project Metadata

Spatial reference system: EPSG:3413 (WGS 84 / NSIDC Sea Ice Polar Stereographic North)
See `configs/project_metadata.yaml` for detailed spatial metadata.
