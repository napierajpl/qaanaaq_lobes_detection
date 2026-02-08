# Script refactoring log

Refactoring scripts following train_model.py approach and .cursorrules (short functions, no long if/elif, extract to utils, fast fail, >200 lines scrutinized). Questions and decisions noted here; resolve doubts after all refactors.

## Script list (by line count, longest first)

| Script | Lines | Status |
|--------|-------|--------|
| train_model.py | 719 | Done (previous refactor) |
| tune_hyperparameters.py | 817 | Done |
| prepare_training_data.py | 348 | Done |
| compute_baseline_metrics.py | 305 | Done |
| analyze_per_tile_performance.py | 272 | Done |
| recalculate_baselines.py | 258 | Done |
| analyze_iou_baseline.py | 172 | Done |
| compare_runs.py | 156 | Done |
| test_model_factory.py | 153 | Done |
| filter_tiles.py | 149 | Done |
| crop_raster.py | 147 | Done |
| create_tile_registry.py | 125 | Done |
| create_tiles.py | 98 | Done |
| resample_raster.py | 94 | Done |
| generate_tile_index_shapefile.py | 89 | Done |
| generate_proximity_map.py | 85 | Done |
| update_proximity_paths.py | 82 | Done |
| rasterize_vector.py | 78 | Done |
| start_mlflow_ui.py | 72 | Pending |
| create_vrt_stack.py | 68 | Done |
| stack_rasters.py | 67 | Done |
| elapsed_time.py | 45 | Pending |

## Questions / decisions

- **tune_hyperparameters.py**: `_current_session_metadata` now uses `path_key` from `data.tile_size` (same as train_model) so session metadata matches the paths actually used for training. Default tile_size=256 kept for path_key to avoid breaking existing configs without `data.tile_size`.
- **All scripts (batch)**: `get_project_root(Path(__file__))` for consistent typing; text file opens use `encoding="utf-8"`. start_mlflow_ui.py and elapsed_time.py not changed (minimal/no get_project_root or file opens).
