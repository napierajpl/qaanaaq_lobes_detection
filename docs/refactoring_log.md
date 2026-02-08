# Script refactoring log

Refactoring scripts following train_model.py approach and .cursorrules (short functions, no long if/elif, extract to utils, fast fail, >200 lines scrutinized). Questions and decisions noted here; resolve doubts after all refactors.

## Script list (by line count, longest first)

| Script | Lines | Status |
|--------|-------|--------|
| train_model.py | 719 | Done (previous refactor) |
| tune_hyperparameters.py | 817 | Done |
| prepare_training_data.py | 348 | Pending |
| compute_baseline_metrics.py | 305 | Pending |
| analyze_per_tile_performance.py | 272 | Pending |
| recalculate_baselines.py | 258 | Pending |
| analyze_iou_baseline.py | 172 | Pending |
| compare_runs.py | 156 | Pending |
| test_model_factory.py | 153 | Pending |
| filter_tiles.py | 149 | Pending |
| crop_raster.py | 147 | Pending |
| create_tile_registry.py | 125 | Pending |
| create_tiles.py | 98 | Pending |
| resample_raster.py | 94 | Pending |
| generate_tile_index_shapefile.py | 89 | Pending |
| generate_proximity_map.py | 85 | Pending |
| update_proximity_paths.py | 82 | Pending |
| rasterize_vector.py | 78 | Pending |
| start_mlflow_ui.py | 72 | Pending |
| create_vrt_stack.py | 68 | Pending |
| stack_rasters.py | 67 | Pending |
| elapsed_time.py | 45 | Pending |

## Questions / decisions

- **tune_hyperparameters.py**: `_current_session_metadata` now uses `path_key` from `data.tile_size` (same as train_model) so session metadata matches the paths actually used for training. Default tile_size=256 kept for path_key to avoid breaking existing configs without `data.tile_size`.
