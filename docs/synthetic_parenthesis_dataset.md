# Synthetic parenthesis dataset (sanity-check)

## Goal

Check that the training pipeline and CNN can learn: same RGB base, add simple target shapes (big black "(" and ")"), predict them. Same architecture, loss, and steps; only the data change. **Synthetic mode is separate from dev/production.** Supports 256 and 512 tile sizes.

## Two generation flows

### 1. From full raster (recommended)

- **Input:** Full RGB `qaanaaq_rgb_0_2m.tif`, boundary vector (valid-data polygons), DEM and slope resampled to match.
- **Boundaries:** By default the script uses `data/raw/vector/research_boundary.shp` (manual boundary from QGIS). Parenthesis are placed **only inside** these polygons (and avoiding white). Alternatively use `-b data/processed/vector/imagery_valid_boundaries.geojson` if you ran `extract_imagery_boundaries.py`.
- **Steps:** Load RGB → place 2000 parenthesis (300 px height) at random inside boundaries → build target raster (0/20) → stack RGB+DEM+Slope → tile to 256 and 512 → write `features/`, `targets/`, `filtered_tiles.json` per tile size.
- **Script:** `scripts/generate_synthetic_parenthesis_from_raster.py`. Requires DEM and slope already resampled (e.g. from `prepare_training_steps.py`).
- **Output:** `data/processed/tiles/synthetic_parenthesis_256/` and `synthetic_parenthesis_512/` (each with features/, targets/, filtered_tiles.json). Full-raster layers (not tiles) under `data/processed/raster/synthetic_parenthesis/`: `synthetic_rgb_with_shapes.tif`, `synthetic_features_5band.tif`, `synthetic_target.tif`. Override with `--raster-output` if needed.

### 2. From existing tiles (legacy)

- **Input:** Existing 256/512 tiles (e.g. dev train) via `filtered_tiles.json` + features dir.
- **Steps:** Per tile, draw parenthesis at random positions, burn into RGB, set target 0/20.
- **Script:** `scripts/generate_synthetic_parenthesis_dataset.py`.
- **Output:** One tile size per run (e.g. `synthetic_parenthesis_512/`).

## Target for learning

Same in both flows: **target raster = 20 inside parenthesis, 0 outside** (same 0–20 range as real proximity; binary). No distance transform.

## Dataset layout

- `data/processed/tiles/synthetic_parenthesis_256/` and `synthetic_parenthesis_512/`
  - `features/` — 5-band GeoTIFFs (RGB + DEM + Slope), RGB has black parenthesis.
  - `targets/` — 1-band GeoTIFFs (0 or 20).
  - `filtered_tiles.json` — list of `{ "tile_id", "features_path", "targets_path" }`.

Training uses the same code path; config points to these dirs via `--mode synthetic_parenthesis`.

## Boundary vector (full-raster flow)

- **Purpose:** Constrain placement (and reuse for other tasks). Parenthesis are drawn only inside the boundary.
- **Default:** `data/raw/vector/research_boundary.shp` (manual boundary from QGIS). Override with `-b` if using another vector.
- **Optional auto-extracted boundaries:** `scripts/extract_imagery_boundaries.py` — from RGB, valid = non-white, vectorize to polygons → `data/processed/vector/imagery_valid_boundaries.geojson`. Use with `-b .../imagery_valid_boundaries.geojson` if preferred.

## Segmentation layer for synthetic

- To train with a 6th (segmentation) channel on synthetic data: create segments from the synthetic RGB-with-shapes layer, then tile to match. Run: `poetry run python scripts/create_segmentation_for_synthetic_parenthesis.py`. It uses `data/processed/raster/synthetic_parenthesis/synthetic_rgb_with_shapes.tif` as input (create that first with `generate_synthetic_parenthesis_from_raster.py`), writes `segmentation_layer.tif` in the same folder, then tiles to `synthetic_parenthesis_256/segmentation/` and `synthetic_parenthesis_512/segmentation/`. Config already has `use_segmentation_layer: true` and `segmentation_dir` for both 256 and 512.

## Segmentation layer (production / other)

- **Separate** from synthetic. Creates a **separate raster layer** (segment IDs) from any input raster (e.g. normal imagery).
- **Script:** `scripts/create_segmentation_layer.py`. Uses Felzenszwalb (OBIA-style). Optional 2 scales (two bands).
- **Limited to boundary:** By default uses `-b data/raw/vector/research_boundary.shp`; pixels outside the boundary are written as nodata. Omit `-b` to segment the full raster.
- **Requires:** `pip install scikit-image` (not in pyproject.toml to avoid dependency conflicts).
- **Use:** Hint for CNN (e.g. extra channel or auxiliary input for boundaries). Does **not** change the training target.

## Limiting training to research boundary

- **Tile filter:** `scripts/filter_tiles_by_boundary.py` — given `filtered_tiles.json` and the research boundary, writes a new tile list containing only tiles that intersect the boundary. Use `--registry` (from `create_tile_registry.py`) for fast bounds, or `--features-dir` so bounds are read from each feature GeoTIFF.
- **Training:** Point your config’s `filtered_tiles` to the output of `filter_tiles_by_boundary.py` (or replace the original file) so training, validation, and tuning only use tiles inside the research boundary.

## Success criterion

If the model quickly reaches low loss and high IoU on this synthetic set, the pipeline and architecture can learn. Then the bottleneck is the real lobe task.
