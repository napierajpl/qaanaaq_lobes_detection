# Script refactoring log

Refactoring scripts following train_model.py approach and .cursorrules (short functions, no long if/elif, extract to utils, fast fail, >200 lines scrutinized). Questions and decisions noted here; resolve doubts after all refactors.

## Script list (by line count, longest first)

| Script | Before | After | +lines | −lines | Affected | Status |
|--------|--------|-------|--------|--------|----------|--------|
| train_model.py | 871 | 719 → 1219 | — | — | — | Second pass planned (see below) |
| tune_hyperparameters.py | 817 | 780 | 51 | 88 | 139 | Done |
| prepare_training_data.py | 342 | 354 | 42 | 30 | 72 | Done |
| compute_baseline_metrics.py | 305 | 305 | 3 | 3 | 6 | Done |
| analyze_per_tile_performance.py | 272 | 272 | 2 | 2 | 4 | Done |
| recalculate_baselines.py | 258 | 258 | 3 | 3 | 6 | Done |
| analyze_iou_baseline.py | 172 | 172 | 2 | 2 | 4 | Done |
| compare_runs.py | 156 | 156 | 2 | 2 | 4 | Done |
| test_model_factory.py | 153 | 153 | 2 | 2 | 4 | Done |
| filter_tiles.py | 149 | 149 | 1 | 1 | 2 | Done |
| crop_raster.py | 147 | 147 | 1 | 1 | 2 | Done |
| create_tile_registry.py | 110 | 125 | 16 | 1 | 17 | Done |
| create_tiles.py | 98 | 98 | 1 | 1 | 2 | Done |
| resample_raster.py | 94 | 94 | 1 | 1 | 2 | Done |
| generate_tile_index_shapefile.py | 76 | 89 | 19 | 6 | 25 | Done |
| generate_proximity_map.py | 85 | 85 | 1 | 1 | 2 | Done |
| update_proximity_paths.py | 82 | 82 | 3 | 3 | 6 | Done |
| rasterize_vector.py | 78 | 78 | 1 | 1 | 2 | Done |
| start_mlflow_ui.py | 72 | 72 | — | — | — | Pending |
| create_vrt_stack.py | 68 | 68 | 1 | 1 | 2 | Done |
| stack_rasters.py | 67 | 67 | 1 | 1 | 2 | Done |
| elapsed_time.py | — | 45 | — | — | — | Pending |

**Note:** *Before* for `train_model.py` is pre–first refactor (commit 95cedbb); for all others, *Before* is pre–batch refactor (commit eb6380f). *Affected* = lines added + lines removed (total lines touched).

## What was done (per script)

- **train_model.py**: Extracted loss factory, config/proximity utils, visualization helpers to `src/`; path_key from `data.tile_size`; MLflow plots path and per-epoch logging; initial placeholder `loss.png`.
- **tune_hyperparameters.py**: (Second pass) Helpers moved to `src/tuning/` (optuna_csv, optuna_session_metadata, optuna_prompts, optuna_best_params). Script is thin orchestration (~419 lines). Handles 0 completed trials without calling `study.best_trial`.
- **prepare_training_data.py**: (Second pass) Step definitions and `PipelineRunner` moved to `src/data_processing/prepare_training_steps.py`; `tile_dir_for_pipeline` in `src/utils/path_utils.py`. Script is thin (~75 lines).
- **compute_baseline_metrics.py**: Config and output file opens with `encoding="utf-8"`.
- **analyze_per_tile_performance.py**: Output file open with `encoding="utf-8"`.
- **recalculate_baselines.py**: Filtered-tiles read/write with `encoding="utf-8"`.
- **analyze_iou_baseline.py**: File opens with `encoding="utf-8"`.
- **compare_runs.py**: Metric file open with `encoding="utf-8"`.
- **test_model_factory.py**: `get_project_root(Path(__file__))`; config open with `encoding="utf-8"`.
- **filter_tiles.py**, **crop_raster.py**, **create_tiles.py**, **resample_raster.py**, **generate_proximity_map.py**, **rasterize_vector.py**, **create_vrt_stack.py**, **stack_rasters.py**: `get_project_root(Path(__file__))` only.
- **create_tile_registry.py**: `get_project_root(Path(__file__))`; added CLI `--tile-size` and `--overlap`, passed into registry creation.
- **generate_tile_index_shapefile.py**: `get_project_root(Path(__file__))`; `--tile-size` with default registry path `train/` or `train_512/`; clearer parser description.
- **update_proximity_paths.py**: `get_project_root(Path(__file__))`; JSON read/write with `encoding="utf-8"`.
- **start_mlflow_ui.py**, **elapsed_time.py**: No refactor (no project root or text file I/O in scope).

## Questions / decisions

- **tune_hyperparameters.py**: `_current_session_metadata` now uses `path_key` from `data.tile_size` (same as train_model) so session metadata matches the paths actually used for training. Default tile_size=256 kept for path_key to avoid breaking existing configs without `data.tile_size`.
- **All scripts (batch)**: `get_project_root(Path(__file__))` for consistent typing; text file opens use `encoding="utf-8"`. start_mlflow_ui.py and elapsed_time.py not changed (minimal/no get_project_root or file opens).

---

## Refactoring plan: train_model.py (second pass)

**Current state:** `scripts/train_model.py` is ~1219 lines. One function `train_model_with_config` is ~960 lines (single responsibility violated; .cursorrules: scrutinize >200 lines, scripts = thin orchestration).

**Goal:** Short functions, one responsibility, OOP-friendly structure; script becomes thin orchestration; business logic under `src/training/`.

### Phase 1 – Resolved config and paths (extract to `src/training/training_config.py`)

- **`resolve_training_paths(config, mode, project_root, filtered_tiles_override)`**  
  Returns a small dataclass or typed dict: `filtered_tiles_path`, `features_dir`, `targets_dir`, `models_dir`, `segmentation_dir`, `slope_stripes_channel_dir`, `path_key`, `tile_size`, `target_mode`, `binary_threshold`.
- **`compute_in_channels(config_data)`**  
  From `use_rgb`, `use_dem`, `use_slope`, `use_segmentation_layer`, `use_slope_stripes_channel` → `in_channels`; raise if 0.
- **`validate_data_splits(train_split, val_split, test_split)`**  
  Check sum == 1.0; raise with clear message otherwise.
- **`apply_illumination_filter(tiles_list, illumination_filter, illumination_include_background)`**  
  Filter train/val/test by illumination; return filtered lists (or same if "all").
- **`get_normalization_stats(train_tiles, features_dir, use_rgb, use_dem, use_slope, use_bg_aug, extended_path)`**  
  Encapsulate “tiles for stats”, `compute_statistics`, and optional logging. Return stats dict or {}.

This keeps path/channel/split/illumination/stats logic out of the giant function and testable.

### Phase 2 – Data loading and model setup (extract to `src/training/` or reuse)

- **`prepare_tiles_and_splits(config, paths, load_filtered_tiles, create_data_splits, max_tiles, viz_config, path_key, tile_size)`**  
  Load filtered tiles, optionally cap with representative + random (max_tiles), split train/val/test, optionally load extended training set, apply illumination filter. Return `(train_tiles, val_tiles, test_tiles, all_tiles)`.
- **`create_training_dataloaders(...)`**  
  Thin wrapper around existing `create_dataloaders` with config-derived args (or keep call in script with args built from Phase 1).
- **`build_model_and_training_components(config, in_channels, target_mode, device)`**  
  Create model (factory), criterion, optimizer, lr_scheduler (or None), early_stopping settings. Return a small container (e.g. dataclass) so the script does not hold 20 lines of scheduler/early-stop setup.

### Phase 3 – MLflow run setup and trial metadata (extract to `src/training/mlflow_run_context.py` or `src/utils/mlflow_utils.py`)

- **`prepare_mlflow_run(run_name, config, mode, ...)`**  
  `setup_mlflow_experiment`, `mlflow.start_run(run_name=...)`, set `loss_plot_path` when tracking is file-based.
- **`log_run_config_and_trial_metadata(active_run, config, trial, applied_best_hparams, ...)`**  
  All `log_training_config`, `log_param`, `set_tag`, and Optuna `trial.set_user_attr` in one place. Keeps script from a 80-line block of logging.
- **`create_initial_loss_placeholder()`**  
  Create and log the initial “Training started” loss figure.

### Phase 4 – Training loop (extract to `src/training/training_loop.py`)

- **`run_one_epoch(epoch, model, train_loader, val_loader, ...)`**  
  Subsample train tiles if needed → rebuild train_loader; unfreeze encoder if epoch matches; `train_one_epoch`; `validate`; return (train_metrics, val_metrics, best_tile_result, best_iou_tile_result).
- **`handle_optuna_pruning(trial, val_loss, epoch, run_name)`**  
  `trial.report`, `should_prune`, print MLflow info, `TrialPruned`.
- **`update_early_stopping(val_loss, ...)`**  
  Update counter and best; return (should_stop, new_counter).
- **`maybe_save_best_and_draw_tiles(...)`**  
  If new best val_loss and throttle passed: save checkpoint; optionally draw best-tile and best-IoU figures and log to MLflow. Return updated `last_time_saved`, `last_drawn_*` state.
- **`run_training_loop(...)`**  
  High-level loop: for epoch in range: run_one_epoch → optuna → lr_scheduler → early_stopping → log_metrics → maybe_save_best_and_draw_tiles → update loss plot; break if early stop. Return (best_val_loss, best_val_mae, best_val_iou, metrics_history, best_tile_info, best_iou_tile_info, ...).

This keeps the script to a single call like `best_val_loss, state = run_training_loop(...)`.

### Phase 5 – Post-training (extract to `src/training/post_training.py` or `visualization.py`)

- **`log_final_metrics_and_trial_attrs(best_val_loss, best_val_mae, best_val_iou, trial)`**  
  MLflow final metrics + trial.set_user_attr.
- **`run_post_training_visualization(config, model, state, paths, ...)`**  
  Load best checkpoint; create full training plots; best-tile and best-IoU figures; representative prediction tiles and channel figures; log all to MLflow. Optionally save to `loss_plot_path` when file tracking.
- **`save_mlflow_model_if_enabled(model, mlflow_config, trial)`**  
  If log_model and not trial: save_model.

### Phase 6 – Script and optional class

- **`train_model_with_config(...)`**  
  Becomes a short sequence: resolve paths (Phase 1) → compute in_channels → validate splits → prepare tiles/splits (Phase 2) → normalization stats → create dataloaders → build model/optimizer/scheduler (Phase 2) → setup MLflow (Phase 3) → run_training_loop (Phase 4) → log_final_metrics (Phase 5) → post_training_visualization (Phase 5) → save_mlflow_model (Phase 5). Return best_val_loss.
- **Optional:** Introduce a `TrainingRun` class in `src/training/run.py` that holds resolved config, paths, model, loaders, and state; methods like `setup()`, `run_loop()`, `finalize()`. Script then does `run = TrainingRun(config, mode, ...); run.setup(); best = run.run_loop(); run.finalize(best)`. Defer class vs plain functions to after Phases 1–5 if we want to avoid big-bang refactor.
- **`main()`**  
  Keep in script: argparse, load config, apply CLI overrides (max_epochs, tile_size, best_hparams, etc.), determine mode, call `train_model_with_config`. Optionally move `_print_applied_hyperparameters` to `src/utils/config_utils.py` or keep in script if only used here.

### Order of work

1. Add `src/training/training_config.py` (Phase 1) and use it from `train_model_with_config` without changing behavior.
2. Add `src/training/training_loop.py` (Phase 4) with small functions; call from script; then extract MLflow run setup (Phase 3) into `src/utils/mlflow_utils.py` or `src/training/mlflow_run_context.py`.
3. Extract post-training (Phase 5) into `src/training/post_training.py` (or extend `visualization.py`).
4. Shrink `train_model_with_config` to the linear sequence above; keep `main()` in script.
5. Re-run tests / smoke run; update this log with final line counts and status.

### Success criteria

- No function in `train_model.py` or new `src/training` modules exceeds ~80–100 lines; most under ~40.
- Script `train_model.py` under ~250 lines (orchestration + main only).
- .cursorrules: short functions, one responsibility, explicit dependencies, no long if/elif in new code.
