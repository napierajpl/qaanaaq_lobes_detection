# Qaanaaq Lobes Detection

CNN pipeline for detecting **solifluction lobes** from high-resolution aerial imagery (0.2 m/pixel) using multi-channel input (RGB + DEM + slope) and U-Net-style models. Outputs proximity maps for lobe locations with MLflow experiment tracking.

**What are solifluction lobes?** Solifluction is the slow downslope flow of water-saturated soil in periglacial regions. Lobes are tongue-shaped or arcuate landforms created by this process; they appear as distinct curved or stepped features on the ground. The data used here comes from **northern Greenland, near the town of Qaanaaq**.

![Solifluction lobes (pink outlines) in the Qaanaaq region, N Greenland — aerial view with contour lines](screenshots/Zrzut%20ekranu%202026-01-23%20173303.png)

*Example from the dataset: barren periglacial terrain with solifluction lobes outlined in pink; contour lines show elevation. Northern Greenland, near Qaanaaq.*

## Setup

**Poetry (recommended)**

```bash
poetry install
poetry shell
```

**Manual**

```bash
pip install -e .
# Optional (SatlasPretrain models): pip install satlaspretrain-models
```

See `pyproject.toml` for full dependencies. Spatial reference: EPSG:3413 (see `configs/project_metadata.yaml`).

**Optional (segmentation layer):** `pip install scikit-image` — required only for `create_segmentation_layer.py` (not in Poetry to avoid dependency conflicts).

---

## Research boundary (AOI)

Many steps can be limited to a **research boundary** (area of interest) so processing and training use only that region.

- **Boundary file:** `data/raw/vector/research_boundary.shp` — create in QGIS (or use another vector). Used by default where applicable.
- **Synthetic parenthesis:** Placement is inside the boundary only (script uses this file by default).
- **Segmentation layer:** Pixels outside the boundary are written as nodata (default: use this boundary).
- **Training:** Filter the tile list to tiles that intersect the boundary, then point config at the filtered list (see [Limiting training to the boundary](#limiting-training-to-the-boundary)).

**Optional – auto-extracted boundaries:** If you prefer valid-data polygons (non-white areas) instead of a manual boundary:

```bash
poetry run python scripts/extract_imagery_boundaries.py -i data/raw/raster/imagery/qaanaaq_rgb_0_2m.tif -o data/processed/vector/imagery_valid_boundaries.geojson
```

Use the output with `-b` in scripts that accept a boundary.

---

## Tile registry

A **tile registry** (`tile_registry.json`) is a single source of truth for tile metadata: geographic bounds, filtering status, train/val/test split, and whether each tile lies inside the research boundary. It is used to limit training to the AOI without opening every GeoTIFF.

**What’s in it**

- **Metadata:** `source_raster`, `tile_size`, `overlap`, `crs`, `created`, `last_updated`.
- **Per tile:** `tile_id`, `tile_idx`, `geographic_bounds` (minx, miny, maxx, maxy), `pixel_bounds` (row/col in source raster), `filtering` (e.g. `is_valid`, `rgb_valid`, `has_targets`), `split` (train / val / test). Optional: `paths` (features/targets), `baseline_metrics`, **`inside_boundary`** (true/false – set when the registry is built or updated with a boundary).

**How it’s created**

- **New registry (train / train_512):**
  `create_tile_registry.py` builds the registry from `filtered_tiles.json` and the source raster (or feature tiles). Use `--boundary data/raw/vector/research_boundary.shp` to set `inside_boundary` for each tile.

  ```bash
  poetry run python scripts/create_tile_registry.py \
    --filtered-tiles data/processed/tiles/train_512/filtered_tiles.json \
    --source-raster data/raw/raster/imagery/qaanaaq_rgb_0_2m.tif \
    --features-dir data/processed/tiles/train_512/features \
    --output data/processed/tiles/train_512/tile_registry.json \
    --tile-size 512 --boundary data/raw/vector/research_boundary.shp
  ```

- **Existing registry (add boundary only):**
  To add or refresh `inside_boundary` without re-running the full migration:

  ```bash
  poetry run python scripts/add_boundary_to_registry.py \
    --registry data/processed/tiles/train_512/tile_registry.json \
    -b data/raw/vector/research_boundary.shp
  ```

- **Synthetic datasets:**
  The full-raster synthetic script (`generate_synthetic_parenthesis_from_raster.py`) writes `tile_registry.json` (with `inside_boundary`) into `synthetic_parenthesis_256/` and `synthetic_parenthesis_512/` automatically.

**How it’s used**

- **`filter_tiles_by_boundary.py`** can use `--registry <path>` so tile bounds come from the registry instead of reading each feature GeoTIFF (faster on large sets).
- You can later limit training or validation to tiles with `inside_boundary: true` by filtering the tile list (e.g. from the registry or from a derived `filtered_tiles.json`).

See [Limiting training to the boundary](#limiting-training-to-the-boundary) for the full flow.

---

## Training

### Production training (full dataset)

1. **Prepare data once** (tiles, proximity maps, filtered tile list):

   ```bash
   poetry run python scripts/prepare_training_data.py
   ```

   Output: `data/processed/tiles/train/` (features, targets, `filtered_tiles.json`).

2. **Optional – extended training set** (background tiles + pre-written augmented lobe tiles). Run after step 1 if you want to use `use_background_and_augmentation: true` in config. Options: use `configs/data_preparation_config.yaml` (augmentation and lobes/background ratio) or pass paths explicitly:

   ```bash
   poetry run python scripts/prepare_extended_training_set.py --config configs/data_preparation_config.yaml
   ```

   For 512×512 tiles, set `tile_size: 512` in the config or run with `--tile-size 512` (uses `paths_512`). Default is 256.

   Or without config (default paths for 256):

   ```bash
   poetry run python scripts/prepare_extended_training_set.py \
     --filtered-tiles data/processed/tiles/train/filtered_tiles.json \
     --features-dir data/processed/tiles/train/features \
     --targets-dir data/processed/tiles/train/targets
   ```

   Output: `features/augmented/`, `targets/augmented/`, and `extended_training_tiles.json` next to `filtered_tiles.json`. Training then loads this JSON when `use_background_and_augmentation: true`.

3. **Run training** (uses `configs/training_config.yaml` and full AOI tiles):

   ```bash
   poetry run python scripts/train_model.py
   ```

   Optional arguments:

   - `--config PATH` – config file (default: `configs/training_config.yaml`)
   - `--run-name NAME` – MLflow run name
   - `--max-epochs N` – override `num_epochs` (e.g. `--max-epochs 1` for a quick run)
   - `--best-hparams` – override config with best hyperparameters from `configs/best_hyperparameters.yaml` (from HP tuning)
   - `--best-hparams-path PATH` – path to best-hparams YAML when using `--best-hparams` (default: `configs/best_hyperparameters.yaml`)
   - `--hp_from_run_id RUN_ID` – apply hyperparameters from an MLflow run (e.g. run ID from MLflow UI); takes precedence over `--best-hparams` if both are set

   If using the extended set, ensure `extended_training_tiles.json` exists (optional step above) and `use_background_and_augmentation: true` in config.

   Runs are logged to MLflow (`./mlruns`). After training, artifacts include loss/MAE/IoU plots and, if configured, prediction-tile visualizations (see `configs/training_config.yaml` → `visualization.representative_tile_ids`).

### Limiting training to the boundary

To train only on tiles that intersect the research boundary:

1. **Create a tile registry** (if you don’t have one) so tile bounds are available. Add `--boundary data/raw/vector/research_boundary.shp` to set `inside_boundary` on each tile (see [Tile registry](#tile-registry)):

   ```bash
   poetry run python scripts/create_tile_registry.py \
     --filtered-tiles data/processed/tiles/train_512/filtered_tiles.json \
     --source-raster data/raw/raster/imagery/qaanaaq_rgb_0_2m.tif \
     --features-dir data/processed/tiles/train_512/features \
     --output data/processed/tiles/train_512/tile_registry.json \
     --tile-size 512 --boundary data/raw/vector/research_boundary.shp
   ```

2. **Filter tiles by boundary:**

   ```bash
   poetry run python scripts/filter_tiles_by_boundary.py \
     --filtered-tiles data/processed/tiles/train_512/filtered_tiles.json \
     -b data/raw/vector/research_boundary.shp \
     --registry data/processed/tiles/train_512/tile_registry.json \
     -o data/processed/tiles/train_512/filtered_tiles_in_boundary.json
   ```

   Without a registry, use `--features-dir data/processed/tiles/train_512/features` instead of `--registry ...` (script will read bounds from each feature GeoTIFF).

3. **Point training at the filtered list:** In `configs/training_config.yaml`, set the path block you use (e.g. `paths.production_512.filtered_tiles`) to `data/processed/tiles/train_512/filtered_tiles_in_boundary.json`, then run `train_model.py` as usual.

### Dev training (small area, fast iteration)

Uses a 1024×1024 cropped area and 36 tiles for quick checks.

1. Prepare dev data:

   ```bash
   poetry run python scripts/prepare_training_data.py --dev
   ```

2. Train:

   ```bash
   poetry run python scripts/train_model.py --dev
   ```

   Optional: `--max-epochs 1` for a single-epoch dry run.

---

## Hyperparameter tuning (Optuna)

Tuning runs multiple training trials with different hyperparameters and can prune poor trials early.

**CLI**

```bash
poetry run python scripts/tune_hyperparameters.py --n-trials 30 [--dev] [--pruning]
```

Common options:

| Option | Default | Description |
|--------|--------|-------------|
| `--config` | `configs/training_config.yaml` | Base training config |
| `--dev` | off | Use dev tiles (faster, smaller dataset) |
| `--n-trials` | 30 | Number of Optuna trials |
| `--study-name` | `lobe_detection_hp_tuning` | Study name (used in storage and MLflow) |
| `--pruning` | on | Enable Optuna pruning |
| `--no-persist` | off | Disable persistent storage (in-memory only) |
| `--results-csv` | `data/optuna_results/<study>_trials.csv` | Where to write trials CSV |
| `--no-seed` | off | Disable seeding first trial from previous best |

Each trial runs training for up to **`num_epochs`** from the config (e.g. 300 in `training_config.yaml`), or until pruning/early stopping. Two **progress plots** are written under `data/optuna_results/` and updated after each trial: (1) **`<study_name>_progress.png`** – trial number vs final value and best-so-far; (2) **`<study_name>_progress_epochs.png`** – validation loss at every epoch across all trials (x-axis labels like T0 E1, T0 E2, …, T1 E1, …). Paths are printed at start (`[PROGRESS] plot_path=...`, `epochs_plot_path=...`); use a viewer that auto-refreshes to follow live.

To **train** (not tune) using hyperparameters from a specific MLflow run (e.g. a past training or tuning run), use **`train_model.py`** with `--hp_from_run_id RUN_ID` (see [Training](#training) optional arguments).

Trials are stored in `data/optuna_studies/<study_name>_<mode>.db` (SQLite) by default. Each trial is logged to MLflow under experiment `lobe_detection_hp_tuning`. Best hyperparameters are written to `configs/best_hyperparameters.yaml` (see script output for path when using custom study name).

**Convenience script**

```bash
./START_HP_TUNING.sh
```

Runs 30 trials with dev tiles and pruning (~8 h order of magnitude). Edit the script to change `--n-trials` or remove `--dev` for production tuning.

---

## Viewing results

Start MLflow UI:

```bash
poetry run python scripts/start_mlflow_ui.py
```

Open http://127.0.0.1:5001. Use it to compare runs, view metrics, and download artifacts (plots, prediction tiles, logged model).

---

## Configuration

- **Training**: `configs/training_config.yaml` – model architecture, loss, optimizer, epochs, data paths, visualization tile IDs.
- **Loss functions**: See [docs/loss_functions.md](docs/loss_functions.md) for descriptions of all options (`smooth_l1`, `weighted_smooth_l1`, `dice`, `iou`, `soft_iou`, `encouragement`, `focal`, `combined`).
- **Best HP (after tuning)**: `configs/best_hyperparameters.yaml` – can be merged or used to update the main config.
- **Spatial/project**: `configs/project_metadata.yaml`.

Architecture options in config: `unet` (baseline) or `satlaspretrain_unet` (recommended; pretrained encoder). See `configs/training_config.yaml` and `docs/model_architecture.md` for details.

---

## Testing

**Unit tests** (fast, no data required):

```bash
poetry run pytest tests/unit -v
```

**End-to-end tests** (minimal data prep, HP tuning, and training on dev data):

- Require dev data: run `poetry run python scripts/prepare_training_data.py --dev` once.
- Run with the `e2e` marker (slow; ~2–3 min for training + tuning):

  ```bash
  poetry run pytest tests/e2e -m e2e -v
  ```

- What they do:
  - **Minimal training**: 1 epoch on dev tiles (256×256), then check MLflow run and `best_val_loss`.
  - **Minimal tuning**: 1 Optuna trial (1 epoch) on dev tiles, then check trials CSV (state and value).
  - **Data prep then training**: run `prepare_training_data --dev` then 1-epoch training (skips if raw data is missing).

Run unit tests with `pytest tests/unit`; use `pytest tests/e2e -m e2e` when you want to run the slow e2e suite.

---

## Synthetic parenthesis dataset (sanity-check)

Quick check that the pipeline can learn: train on synthetic shapes (black “(” and “)” on real imagery) instead of lobes.

**From full raster (recommended)** — uses `data/raw/vector/research_boundary.shp` so shapes are placed only inside the boundary:

1. Ensure DEM and slope are resampled (e.g. already done by `prepare_training_data.py` or `prepare_training_steps.py`).
2. Generate synthetic tiles (256 and 512):

   ```bash
   poetry run python scripts/generate_synthetic_parenthesis_from_raster.py
   ```

   Output: `data/processed/tiles/synthetic_parenthesis_256/` and `synthetic_parenthesis_512/` (features, targets, `filtered_tiles.json`). Override boundary with `-b path/to/vector.shp` or use `-b data/processed/vector/imagery_valid_boundaries.geojson` if you ran `extract_imagery_boundaries.py`.

3. Train in synthetic mode:

   ```bash
   poetry run python scripts/train_model.py --config configs/training_config_synthetic_parenthesis.yaml --mode synthetic_parenthesis
   ```

See `docs/synthetic_parenthesis_dataset.md` for the legacy tile-based generator and details.

---

## Segmentation layer (optional)

A **separate raster layer** of segment IDs (OBIA-style, Felzenszwalb) can be used as a **6th input channel** to the CNN for boundary hints. By default it is limited to the research boundary (nodata outside).

- **Requires:** `pip install scikit-image`
- **1) Create the full raster** (default: input = full RGB, output = `data/processed/raster/imagery_segmentation_layer.tif`, boundary = research_boundary.shp):

  ```bash
  poetry run python scripts/create_segmentation_layer.py
  ```

  Options: `-i` / `-o` for input/output raster, `-b` for boundary (omit to segment full raster), `--scale` / `--scale2` for segment size, `--block-size` for large rasters. See `scripts/create_segmentation_layer.py --help`.

- **2) Tile the segmentation raster** with the same tile size and overlap as your feature tiles (e.g. 512×512, 30% overlap), so each feature tile has a matching segmentation tile:

  ```bash
  poetry run python scripts/create_tiles.py -i data/processed/raster/imagery_segmentation_layer.tif -o data/processed/tiles/train_512/segmentation --tile-size 512 --overlap 0.3 --no-organize
  ```

- **3) Enable in training:** In `configs/training_config.yaml` set `data.use_segmentation_layer: true` and under the chosen path (e.g. `paths.production_512`) set `segmentation_dir: "data/processed/tiles/train_512/segmentation"`. The model will use 6 input channels (RGB + DEM + slope + segmentation).

---

## Other scripts

- **Data**: `rasterize_vector.py`, `generate_proximity_map.py`, `create_tiles.py`, `filter_tiles.py` – building blocks; usually run via `prepare_training_data.py`.
- **Boundary / AOI**: `extract_imagery_boundaries.py` – vectorize valid-data (non-white) regions to GeoJSON; `filter_tiles_by_boundary.py` – filter `filtered_tiles.json` to tiles intersecting a boundary (e.g. research_boundary.shp).
- **Synthetic**: `generate_synthetic_parenthesis_from_raster.py` – full-raster synthetic parenthesis inside boundary, then tile to 256/512; `generate_synthetic_parenthesis_dataset.py` – legacy tile-based synthetic data.
- **Segmentation**: `create_segmentation_layer.py` – OBIA-style segment ID raster (optional CNN hint), limited to boundary by default.
- **Analysis**: `compute_baseline_metrics.py`, `analyze_per_tile_performance.py`, `compare_runs.py`.
- **QGIS**: `generate_tile_index_shapefile.py` – tile index shapefile for the map; use `--tile-size 512` for 512×512 tiles (requires a registry in `train_512/`).

See `docs/PROJECT_STRUCTURE.md` for folder layout and `docs/` for guides (e.g. `training_how_it_works.md`, `training_visualization.md`, `OPTUNA_QUICK_START.md`, `synthetic_parenthesis_dataset.md`, `plan_synthetic_parenthesis_and_multiscale.md`).
