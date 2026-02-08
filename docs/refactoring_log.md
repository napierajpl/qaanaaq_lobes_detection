# Script refactoring log

Refactoring scripts following train_model.py approach and .cursorrules (short functions, no long if/elif, extract to utils, fast fail, >200 lines scrutinized). Questions and decisions noted here; resolve doubts after all refactors.

## Script list (by line count, longest first)

| Script | Before | After | +lines | −lines | Affected | Status |
|--------|--------|-------|--------|--------|----------|--------|
| train_model.py | 871 | 719 | 88 | 240 | 328 | Done (previous refactor) |
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
