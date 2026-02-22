# Workflow: 512×512 tiles from scratch to training

Order of operations (you didn’t miss anything):

1. **Create 512 tiles** (and filtered list)
2. **Create tile registry** (needed for shapefile)
3. **Generate shapefile** (optional, for QGIS)
4. **Choose tiles for illustration** (for MLflow prediction figures)
5. **Set config** and **train**

---

## 1. Create 512 tiles

This builds the VRT stack and proximity map (if needed), then tiles and filters.

**Production (full AOI):**
```bash
poetry run python scripts/prepare_training_data.py --tile-size 512
```
Output: `data/processed/tiles/train_512/features/`, `targets/`, `filtered_tiles.json`.

**Dev (1024×1024 crop, quick check):**
```bash
poetry run python scripts/prepare_training_data.py --dev --tile-size 512
```
Output: `data/processed/tiles/dev/train_512/...`

---

## 2. Create tile registry

Needed for the shapefile and as a single source of tile metadata.

**Production:**
```bash
poetry run python scripts/create_tile_registry.py \
  --filtered-tiles data/processed/tiles/train_512/filtered_tiles.json \
  --features-dir data/processed/tiles/train_512/features \
  --output data/processed/tiles/train_512/tile_registry.json \
  --tile-size 512 --overlap 0.3
```

**Dev:** use `data/processed/tiles/dev/train_512/...` for the paths above.

---

## 3. Generate shapefile (optional, for QGIS)

```bash
poetry run python scripts/generate_tile_index_shapefile.py --tile-size 512
```
Output: `data/processed/tiles/train_512/tile_index.shp` (+ `.qml`).
Load in QGIS to inspect the 512 grid.

---

## 4. Choose tiles for illustration

- **Option A:** In QGIS, open `tile_index.shp`, pick a few `tile_id` or `tile_idx` values you want as MLflow prediction figures.
- **Option B:** Open `data/processed/tiles/train_512/filtered_tiles.json`, check `stats.valid_tiles` and pick a few indices (e.g. `0`, `100`, `500`) that exist in `tiles`.

Then set in `configs/training_config.yaml` under `visualization`:
```yaml
representative_tile_ids_512: [0, 100, 500]   # your chosen indices or tile_ids
```
Leave `[]` if you don’t want prediction-tile artifacts.

---

## 5. Set config and train

In `configs/training_config.yaml`:

- Set **`data.tile_size: 512`** (so training uses `paths.production_512` and 512×512 tiles).
- Optionally in `best_hyperparameters.yaml` set **`tile_size: 512`** if you train with `--best-hparams`.

**Note:** 512×512 tiles use ~4× memory per sample. If you hit OOM, reduce `training.batch_size` (e.g. from 16 to 8 or 4).

Then run training:
```bash
poetry run python scripts/train_model.py
```
With dev data:
```bash
poetry run python scripts/train_model.py --dev
```

---

## Summary checklist

| Step | Command / action |
|------|-------------------|
| 1. Tiles | `prepare_training_data.py --tile-size 512` (or `--dev --tile-size 512`) |
| 2. Registry | `create_tile_registry.py` with `train_512` paths and `--tile-size 512` |
| 3. Shapefile | `generate_tile_index_shapefile.py --tile-size 512` |
| 4. Illustration | Set `visualization.representative_tile_ids_512` in config (or leave `[]`) |
| 5. Train | `data.tile_size: 512` in config, then `train_model.py` (reduce batch_size if OOM) |
