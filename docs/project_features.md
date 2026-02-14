# Project Features — Full Inventory

Single source of truth for what the lobe detection project supports: models, losses, tuning, visualization, MLflow, data preparation, and which backlog items are already implemented. Last updated: 2026-02-14.

---

## 1. Models / Architectures

**Purpose:** Define which network architectures and encoders can be used so we can switch between baseline and pretrained backbones from config.

**Model factory:** `src/models/factory.py` — creates model from `config["model"]`.

| Architecture | Config key | Description |
|-------------|------------|-------------|
| **U-Net** | `unet` | Baseline U-Net, no pretrained encoder. Config: `in_channels`, `out_channels`, `base_channels`, `dropout`. |
| **SatlasPretrain U-Net** | `satlaspretrain_unet` | U-Net with SatlasPretrain pretrained encoder. Config: `encoder.name`, `encoder.pretrained`, `encoder.freeze_encoder`, `encoder.unfreeze_after_epoch`, `decoder_dropout`. Optional **SE + PPM** (Gully-ERFNet style): `use_se`, `use_ppm` (config-only; supports ablation: baseline, +SE, +PPM, +both). |

**Encoder options (SatlasPretrain only):** `resnet50`, `resnet152`, `swin_v2_base`, `swin_v2_tiny`.

**Input:** 5 channels (RGB + DEM + Slope). SatlasPretrain uses a 5→3 channel adapter for the pretrained backbone.

**Sources:** `src/models/architectures.py` (UNet), `src/models/satlaspretrain_unet.py` (SatlasPretrainUNet).

---

## 2. Loss Functions

**Purpose:** Provide multiple loss options (regression, segmentation, combined) for proximity-map training and handle class imbalance; choice is config-driven.

**Factory:** `src/training/loss_factory.py` — `create_criterion(training_config)`; config key: `training.loss_function`.

| Loss | Config key | Main config parameters |
|------|------------|-------------------------|
| **Smooth L1** | `smooth_l1` | — |
| **Weighted Smooth L1** | `weighted_smooth_l1` | `lobe_weight`, `iou_threshold` |
| **Dice** | `dice` | `iou_threshold` |
| **IoU** | `iou` | `iou_threshold` |
| **Soft IoU** | `soft_iou` | `iou_threshold` |
| **Encouragement** | `encouragement` | `iou_threshold`, `encouragement_weight` |
| **Focal** | `focal` | `focal_alpha`, `focal_gamma`, `iou_threshold` |
| **Combined** | `combined` | `iou_weight`, `regression_weight`, `iou_threshold`, `lobe_weight`, `use_soft_iou` |
| **ACL (Adaptive Correction)** | `acl` | `acl_lambda`, `iou_threshold`, `focal_alpha`, `focal_gamma` |

**Shared:** `iou_threshold` — pixels with value ≥ this are treated as lobe (used for binarization in IoU/Dice/Focal/ACL, etc.). Proximity maps are 0–20; config often uses 1.0 or 5.0.

**Implementations:** `src/models/losses.py`. See also `docs/loss_functions.md`.

---

## 3. Hyperparameter Tuning (Optuna)

**Purpose:** Automate search over learning rate, batch size, loss, encoder, and related knobs so we can find better configs without manual sweeps; results feed into best-hparams and CSV history.

**Script:** `scripts/tune_hyperparameters.py`. Uses `scripts/train_model.train_model_with_config` as objective.

### 3.1 Tuned parameters (search space)

| Parameter | Type | Range / choices |
|-----------|------|------------------|
| `learning_rate` | float (log) | 1e-4 – 1e-1 |
| `batch_size` | categorical | [8, 16, 32, 64] |
| `weight_decay` | float (log) | 1e-5 – 1e-2 |
| `focal_alpha` | float | 0.25 – 0.95 |
| `focal_gamma` | float | 1.0 – 4.0 |
| `loss_function` | categorical | `focal`, `combined`, `weighted_smooth_l1` |
| `encoder_name` | categorical | `resnet50`, `resnet152`, `swin_v2_base`, `swin_v2_tiny` |
| `decoder_dropout` | float | 0.0 – 0.5 |
| `lr_scheduler_patience` | int | 5 – 20 |
| `lr_scheduler_factor` | float | 0.1 – 0.9 |
| `max_grad_norm` | float (log) | 0.1 – 10.0 |
| `unfreeze_after_epoch` | int | 0 – 50 |

**Objective:** Minimize best validation loss. Pruning: `trial.report(val_loss, epoch)` each epoch; `MedianPruner` (optional via `--pruning`).

### 3.2 Config mapping

Best-HP → config paths: `src/utils/config_utils.py` — `BEST_HP_CONFIG_PATHS` (e.g. `learning_rate` → `training.learning_rate`, `encoder_name` → `model.encoder.name`). Applied when using `--best-hparams` or `--hp_from_run_id` in `train_model.py`.

### 3.3 Tuning workflow

- **Sampler:** TPESampler (seed=42).
- **Pruner:** MedianPruner when `--pruning`.
- **Seed trials:** Can enqueue a previous best trial from CSV (`--import-from`); prompts for “use recent” vs “use compatible” when compatibility differs.
- **Output:** Best params saved to `configs/best_hyperparameters.yaml`; trials appended to CSV (e.g. `data/optuna_results/<study_name>_trials.csv`).
- **MLflow:** Tuning runs use experiment `lobe_detection_hp_tuning`; each trial is one MLflow run.

**Support modules:** `src/tuning/optuna_csv.py` (CSV load/append, compatibility, seed enqueue), `src/tuning/optuna_best_params.py` (save best to YAML), `src/tuning/optuna_session_metadata.py`, `src/tuning/optuna_prompts.py`, `src/tuning/optuna_plots.py` (progress plot update).

---

## 4. Training Configuration (non-tuning)

**Purpose:** Centralize all training knobs (optimizer, scheduler, early stopping, data paths, splits) in one config so runs are reproducible and easy to override via CLI or best-hparams.

**Config file:** `configs/training_config.yaml`.

- **Optimizer:** Adam; `learning_rate`, `weight_decay`.
- **LR scheduler:** ReduceLROnPlateau; `lr_scheduler_patience`, `lr_scheduler_factor`, `lr_scheduler_min_lr`.
- **Early stopping:** `early_stopping_patience`, `early_stopping_min_delta`.
- **Gradient clipping:** `max_grad_norm`.
- **Data:** `tile_size` (256 or 512), `train_split` / `val_split` / `test_split`, `normalize_rgb`, `standardize_dem`, `standardize_slope`, `use_background_and_augmentation`, `train_subsample_ratio`.
- **Paths:** Per mode (`dev`, `dev_512`, `production`, `production_512`: `filtered_tiles`, `features_dir`, `targets_dir`, `models_dir`).

**CLI (train_model.py):** `--config`, `--dev`, `--run-name`, `--max-epochs`, `--best-hparams`, `--best-hparams-path`, `--hp_from_run_id`.

---

## 5. Visualization

**Purpose:** Produce training curves and per-tile prediction figures for quick sanity checks and comparison to baselines; all plots can be logged as MLflow artifacts.

**Module:** `src/training/visualization.py`. Used from `scripts/train_model.py` and tuning.

| Feature | Function | Description |
|--------|----------|-------------|
| **Loss curve** | `plot_loss` | Train vs val loss by epoch. |
| **MAE comparison** | `plot_mae_comparison` | Model MAE vs baseline MAE (horizontal line). |
| **IoU curve** | `plot_iou` | Validation IoU by epoch. |
| **Improvement %** | `plot_improvement_percentage` | Percent improvement over baseline by epoch. |
| **All training plots** | `create_training_plots` | Builds loss, mae_comparison, iou, improvement_percent; optional `output_dir` to save. |
| **Representative-tile figures** | `create_prediction_tile_figures` | Per-tile figures: RGB, proximity (target), prediction; optional per-tile MAE/RMSE/IoU. |
| **Best predicted tile** | `show_best_predicted_tile` | Single figure for the validation tile with lowest loss. |
| **Tile ID resolution** | `get_representative_tile_ids_for_viz`, `resolve_representative_tiles` | Config → list of tile indices/IDs; resolve to tile dicts for current dataset. |

**Config:** `configs/training_config.yaml` → `visualization`: `representative_tile_ids`, `representative_tile_ids_512`, `representative_tile_ids_dev_512`, `prediction_tiles_fallback_n`. Figures saved as MLflow artifacts (e.g. `plots/loss.png`, `prediction_tiles/<tile_id>.png`).

---

## 6. MLflow

**Purpose:** Track experiments, log config and metrics, save model and plot artifacts, and compare runs so we can reproduce and compare training runs without ad-hoc scripts.

**Module:** `src/utils/mlflow_utils.py`. **UI:** `scripts/start_mlflow_ui.py`.

| Feature | Description |
|--------|-------------|
| **Experiment setup** | `setup_mlflow_experiment(experiment_name, tracking_uri)`. Default tracking: `file:./mlruns`. |
| **Config logging** | `log_training_config(config)` — flattened params. |
| **Metrics** | `log_metrics(metrics, step)` — e.g. train/val loss, MAE, IoU, baseline comparison, improvement %. |
| **Model artifact** | `save_model(model, artifact_path)` — PyTorch model logged; model size (MB/bytes) logged as params. |
| **Artifacts** | Training plots (loss, mae_comparison, iou, improvement_percent), prediction tile figures, best checkpoint path. |

**Config:** `configs/training_config.yaml` → `mlflow`: `experiment_name`, `tracking_uri`, `log_artifacts`, `log_model`. Training uses `lobe_detection`; tuning uses `lobe_detection_hp_tuning`.

**Run comparison:** `scripts/compare_runs.py` — compare two MLflow runs (params + final metrics) by run ID.

---

## 7. Data Preparation

**Purpose:** Turn raw rasters and vector lobes into tiled features/targets and filtered tile lists (with optional baselines), and optionally build an extended set with background tiles and augmentation.

### 7.1 Main pipeline (production or dev)

**Script:** `scripts/prepare_training_data.py`. Uses `src/data_processing/prepare_training_steps.py` (`PipelineRunner`, `production_steps`, `dev_steps`).

**Production steps (high level):**

1. Generate proximity map (20px) from rasterized lobes.
2. Resample DEM and slope to match RGB.
3. Create VRT stack: RGB + DEM + slope.
4. Create feature tiles (e.g. 256×256 or 512×512, 30% overlap).
5. Create target tiles (same grid, proximity map).
6. Filter tiles and compute baselines (`filter_tiles.py --exclude-background`, lobe threshold 5.0).

**Dev steps:** Same idea on cropped 1024×1024 inputs, then tile and filter.

**Scripts involved:** `generate_proximity_map.py`, `resample_raster.py`, `create_vrt_stack.py`, `create_tiles.py`, `filter_tiles.py`, `crop_raster.py` (dev).

### 7.2 Extended training set (background + augmentation)

**Script:** `scripts/prepare_extended_training_set.py`. **Config:** `configs/data_preparation_config.yaml`.

- **Background tiles:** From excluded (non-lobe) tiles; filter by `white_threshold`; ratio to (lobe + augmented lobe) set via `background.ratio`.
- **Augmentation (lobe tiles):** Rotations (0°, 90°, 180°, 270°), contrast/saturation ranges; optional cap `n_lobe_tiles_to_augment`.
- **Output:** `extended_training_tiles.json` (path from config or same dir as `filtered_tiles`). Separate runs for 256 vs 512 (`tile_size` in config).

**Data loading:** If `data.use_background_and_augmentation` is true, training uses `load_extended_training_tiles` and `build_extended_train_tiles`; otherwise `load_filtered_tiles` and standard splits.

### 7.3 Tile filtering and baselines

**Module:** `src/data_processing/tile_filter.py` — `TileFilter`.

- **RGB check:** Exclude tiles that are effectively all-white (configurable threshold).
- **Target check:** Lobe presence (e.g. any value ≥ lobe_threshold); optional `compute_baselines=True`.
- **Baselines:** Per-tile baseline metrics (e.g. predict-zero MAE/RMSE/IoU, per-class baselines). Stored in `filtered_tiles.json` under `target_stats.baseline_metrics`.

**Script:** `scripts/filter_tiles.py` (CLI for features/targets dirs, output JSON, `--exclude-background`, `--lobe-threshold`).

Other scripts: `recalculate_baselines.py`, `compute_baseline_metrics.py`, `analyze_iou_baseline.py`, `analyze_per_tile_performance.py`.

### 7.4 Raster and vector utilities

- **Proximity:** `src/data_processing/raster_utils.py` — `ProximityMapGenerator` (max_value, max_distance), `generate_proximity_map()`.
- **Raster cropping:** `RasterCropper`, `crop_raster()`; **resampling:** `resample_raster_to_match`.
- **VRT stack:** `VirtualRasterStacker`, `create_vrt_stack()`.
- **Vector → raster:** `src/data_processing/vector_utils.py` — `Rasterizer`, `rasterize_vector()`.
- **Tiling:** `src/data_processing/tiling.py` — `Tiler` (tile_size, overlap), `tile_raster()`, `create_tile_filename()`.

---

## 8. Data Loading and Preprocessing

**Purpose:** Load tiles with consistent normalization/standardization and optional augmentation, and build train/val/test DataLoaders (including extended set and subsampling).

**Module:** `src/training/dataloader.py`.

- **TileDataset:** Loads (features, target) per tile; applies normalization/standardization; optional photometric augmentation (contrast/saturation on RGB only) for extended set.
- **Normalization:** `src/preprocessing/normalization.py` — `normalize_rgb` (0–255 → 0–1), `standardize_dem`, `standardize_slope` (mean=0, std=1); `compute_statistics(tile_paths)` for DEM/slope stats from tiles.
- **Splits:** `create_data_splits(tiles, train_split, val_split, test_split, seed)`; `create_dataloaders(...)` for train/val/test DataLoaders.
- **Train subsampling:** Config `train_subsample_ratio` (0.0–1.0) — each epoch uses a random subset of train tiles.

---

## 9. Evaluation Metrics

**Purpose:** Define the metrics (MAE, RMSE, IoU) used for training feedback, validation, and per-tile visualization so all components use the same definitions.

**Module:** `src/evaluation/metrics.py`.

| Metric | Function | Note |
|--------|----------|------|
| **MAE** | `compute_mae(pred, target)` | Mean absolute error. |
| **RMSE** | `compute_rmse(pred, target)` | Root mean squared error. |
| **IoU** | `compute_iou(pred, target, threshold)` | Binary IoU; pixels ≥ `threshold` are positive. |

Used in `src/training/trainer.py` (train/val) and in visualization (per-tile MAE/RMSE/IoU).

---

## 10. Tile Registry and Map Overlays

**Purpose:** Maintain a single registry of tile metadata (bounds, splits, baselines, model metrics) and export shapefiles + QML for QGIS so we can inspect coverage and performance geographically.

**Tile registry:** `src/map_overlays/tile_registry.py` — `TileRegistry(registry_path, source_raster_path)`.

- Load/save registry; migrate from `filtered_tiles.json`; geographic bounds, filtering status, splits, baseline metrics.
- **Update model metrics:** `update_model_metrics(tile_id, mae, rmse, iou, improvement_over_baseline)`.
- **Queries:** `get_tile(tile_id)`, `get_all_tiles(filter_valid, filter_split)`.

**Script:** `scripts/create_tile_registry.py` — build registry from filtered tiles, optional source raster, train/val/test split fractions.

**Shapefile for QGIS:** `src/map_overlays/shapefile_generator.py` — `generate_tile_index_shapefile(registry, output_path, label_field, include_all_tiles, background_train_ids)`. Generates .shp + .qml (by split or by train_usage). **Script:** `scripts/generate_tile_index_shapefile.py` (uses registry + config for paths).

---

## 11. Pipelines

**Purpose:** Run the full workflow (data prep → extended set → tuning → training) in one script with optional skips so we can reproduce end-to-end runs or only re-train.

**Full pipeline:** `pipelines/run_full_pipeline.sh`.

1. Data preparation: `prepare_training_data.py` (optional `--dev`).
2. Extended training set: `prepare_extended_training_set.py --config configs/data_preparation_config.yaml` (skippable with `--skip-extended`).
3. Hyperparameter tuning: `tune_hyperparameters.py --n-trials N --pruning` (skippable with `--skip-tuning`).
4. Training: `train_model.py` with or without `--best-hparams` depending on whether tuning was run.

Options: `--dev`, `--n-trials N`, `--skip-extended`, `--skip-tuning`.

---

## 12. Other Scripts (summary)

**Purpose:** Quick reference for what each script does so we know the right entry point for data prep, training, tuning, or analysis.

| Script | Purpose |
|--------|---------|
| `train_model.py` | Train with config; optional best-HP or MLflow run ID. |
| `tune_hyperparameters.py` | Optuna HP search; CSV export; best params YAML. |
| `prepare_training_data.py` | Run dev or production data-prep steps. |
| `prepare_extended_training_set.py` | Build extended_training_tiles.json (background + augmentation). |
| `filter_tiles.py` | Filter feature/target pairs; write filtered_tiles.json; optional baselines. |
| `create_tile_registry.py` | Build tile registry from filtered_tiles.json. |
| `generate_tile_index_shapefile.py` | Export tile index shapefile + QML for QGIS. |
| `generate_proximity_map.py` | Proximity transform from binary lobe raster. |
| `create_tiles.py` | Tile rasters (features/targets) with overlap. |
| `create_vrt_stack.py` | Stack rasters into VRT. |
| `crop_raster.py` | Crop raster by geo window. |
| `resample_raster.py` | Resample raster to match reference. |
| `rasterize_vector.py` | Rasterize vector to georeferenced raster. |
| `update_proximity_paths.py` | Update paths to proximity raster in config/data. |
| `stack_rasters.py` | Stack rasters (alternative to VRT). |
| `compare_runs.py` | Compare two MLflow runs. |
| `start_mlflow_ui.py` | Launch MLflow UI. |
| `analyze_iou_baseline.py` | Analyze IoU vs baseline. |
| `analyze_per_tile_performance.py` | Per-tile performance analysis. |
| `recalculate_baselines.py` | Recompute baseline metrics. |
| `compute_baseline_metrics.py` | Compute baseline metrics. |
| `elapsed_time.py` | Utility for elapsed time. |
| `test_model_factory.py` | Test model creation. |

---

## 13. Config Files

**Purpose:** List the main config files and what they control so we know where to change model, training, data, or MLflow settings.

| File | Purpose |
|------|----------|
| `configs/training_config.yaml` | Model, training, data, visualization, MLflow, paths. |
| `configs/best_hyperparameters.yaml` | Best validation loss, best trial number, hyperparameters (written by Optuna). |
| `configs/data_preparation_config.yaml` | Extended set: paths (256/512), splits, background ratio, augmentation. |
| `configs/project_metadata.yaml` | Project metadata. |

---

## 14. Implemented Features (from backlog)

**Purpose:** Record which improvements from `docs/improvements_backlog.md` are already done so we avoid re-implementing and can see what is in place.

Items below were moved here from the backlog after verification in the codebase.

### Architecture

- **Pretrained encoder (SatlasPretrain)** — Option B from backlog item 10.
  - SatlasPretrain ResNet50 (and optionally other encoders) trained on 302M remote sensing labels.
  - Implementation: `src/models/satlaspretrain_unet.py`, `src/models/factory.py`.
  - 5-channel input adapter (RGB + DEM + Slope) for 3-channel pretrained encoder.
  - Encoder freeze/unfreeze support (e.g. freeze initially, fine-tune after convergence).
  - Model factory for architecture switching (`unet` vs `satlaspretrain_unet`).
  - Support for ResNet50, ResNet152, Swin-v2-Base, Swin-v2-Tiny.
  - Plans: `docs/implementation_plans/satlaspretrain_integration.md`, `docs/implementation_plans/satlaspretrain_implementation_summary.md`.
- **SE + Pyramid Pooling Module (PPM)** — Backlog item 5.
  - Optional Squeeze-and-Excitation (SE) and Pyramid Pooling Module (PPM) on deepest encoder feature (enc4) before bottleneck. Config: `model.use_se`, `model.use_ppm` (flat); optional `ppm_bins`, `se_reduction`. Modules created only when enabled; ablation: baseline (both false), +SE, +PPM, +both. Implementation: `src/models/se_ppm.py`, integrated in `SatlasPretrainUNet`. Not in Optuna search space.

### Training infrastructure

- **Dropout in decoder** — 0.2 in decoder blocks (U-Net and SatlasPretrain U-Net). Config: `decoder_dropout`.
- **Gradient clipping** — `torch.nn.utils.clip_grad_norm_` in `src/training/trainer.py`; config: `training.max_grad_norm`.
- **Learning rate scheduling** — ReduceLROnPlateau in `scripts/train_model.py`; config: `lr_scheduler_patience`, `lr_scheduler_factor`.
- **Early stopping** — Patience-based on validation loss in `scripts/train_model.py`; config: `early_stopping_patience`, `early_stopping_min_delta`.

### Evaluation and visualization

- **Per-tile baseline comparison** — Baseline MAE (e.g. predict-zero) per tile in `src/data_processing/tile_filter.py`; validation metrics vs baseline in `src/training/trainer.py`; improvement-over-baseline plots in `src/training/visualization.py`.
- **Training visualization module** — Loss/MAE/IoU plots, MAE vs baseline, percent improvement over baseline; `src/training/visualization.py`, used from `scripts/train_model.py`.
- **MLflow integration** — Experiments, runs, config logging, metrics, artifacts (plots, checkpoints); `src/utils/mlflow_utils.py`, `scripts/train_model.py`, `scripts/start_mlflow_ui.py`.
- **Representative-tile prediction visualization** — After each training run (non-Optuna), MLflow artifacts include prediction figures for a configurable list of tiles (RGB, proximity target, prediction). Config: `configs/training_config.yaml` → `visualization.representative_tile_ids` (and dev/512 variants). Full mosaic/GeoTIFF export is not implemented and remains in the backlog.

### Research / literature

- **Literature search: similar tasks (linear structures, imbalance)** — Backlog item 18.
  - Deliverable: `docs/literature/similar_tasks_imbalance_linear_structures.md`.
  - Content: 4 papers (ResUNet-a, Boundary-Aware U-Net glacier, UFL, Gully-ERFNet); priorities in backlog aligned with “Next steps.”
