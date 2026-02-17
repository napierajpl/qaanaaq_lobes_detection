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

## Other scripts

- **Data**: `rasterize_vector.py`, `generate_proximity_map.py`, `create_tiles.py`, `filter_tiles.py` – building blocks; usually run via `prepare_training_data.py`.
- **Analysis**: `compute_baseline_metrics.py`, `analyze_per_tile_performance.py`, `compare_runs.py`.
- **QGIS**: `generate_tile_index_shapefile.py` – tile index shapefile for the map; use `--tile-size 512` for 512×512 tiles (requires a registry in `train_512/`).

See `docs/PROJECT_STRUCTURE.md` for folder layout and `docs/` for guides (e.g. `training_how_it_works.md`, `training_visualization.md`, `OPTUNA_QUICK_START.md`).
