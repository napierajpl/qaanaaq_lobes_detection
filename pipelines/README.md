# Pipelines

Bash workflows for data preparation and training. Run from **project root**: `bash pipelines/<script>.sh [OPTIONS]`.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `run_full_pipeline.sh` | Generic: data prep → extended set → optional tuning → training. Use `--dev` for dev. |
| `run_dev_pipeline.sh` | Dev workflow: data prep (256 default) → extended → tuning → train with `--dev`. |
| `run_production_pipeline.sh` | Production workflow: data prep (512) → extended → tuning → train. |
| `run_synthetic_parenthesis_pipeline.sh` | Sanity-check: generate synthetic 512 dataset → optional baselines → train `--mode synthetic_parenthesis`. |

---

## Steps to start training

### Dev

1. Prepare dev data (256 or 512 tiles):
   ```bash
   poetry run python scripts/prepare_training_data.py --dev --tile-size 256
   ```
2. (Optional) Extended set: `poetry run python scripts/prepare_extended_training_set.py --config configs/data_preparation_config.yaml`
3. (Optional) Recalculate baselines so training logs `improvement_over_baseline`:
   ```bash
   poetry run python scripts/recalculate_baselines.py --input data/processed/tiles/dev/train/filtered_tiles.json --targets-dir data/processed/tiles/dev/train/targets
   ```
   For 512 dev: `--input data/processed/tiles/dev/train_512/filtered_tiles.json` and `--targets-dir data/processed/tiles/dev/train_512/targets`
4. Train:
   ```bash
   poetry run python scripts/train_model.py --dev --tile-size 256
   ```

Or run the dev pipeline:

```bash
bash pipelines/run_dev_pipeline.sh [--skip-extended] [--skip-tuning] [--tile-size 256|512]
```

### Production

1. Prepare production data (512):
   ```bash
   poetry run python scripts/prepare_training_data.py --tile-size 512
   ```
2. (Optional) Extended set and recalculate baselines:
   ```bash
   poetry run python scripts/recalculate_baselines.py --input data/processed/tiles/train_512/filtered_tiles.json --targets-dir data/processed/tiles/train_512/targets
   ```
3. Train:
   ```bash
   poetry run python scripts/train_model.py
   ```

Or:

```bash
bash pipelines/run_production_pipeline.sh [--skip-extended] [--skip-tuning]
```

### Synthetic parenthesis (sanity-check, 512 only)

1. **Prerequisite**: Source 512 tiles must exist (e.g. run dev data prep with 512 first):
   ```bash
   poetry run python scripts/prepare_training_data.py --dev --tile-size 512
   ```
2. Generate dataset and train:
   ```bash
   bash pipelines/run_synthetic_parenthesis_pipeline.sh
   ```
   Options: `--skip-baselines`, `--max-tiles N`, `--source dev|prod`.

Or manually:

```bash
poetry run python scripts/generate_synthetic_parenthesis_dataset.py
poetry run python scripts/recalculate_baselines.py --input data/processed/tiles/synthetic_parenthesis_512/filtered_tiles.json --targets-dir data/processed/tiles/synthetic_parenthesis_512/targets --lobe-threshold 1.0
poetry run python scripts/train_model.py --mode synthetic_parenthesis
```

---

## Baselines for new mode

- **Dev / production**: Run `recalculate_baselines.py` with the correct `--input` (filtered_tiles.json) and `--targets-dir` for that dataset. Training uses these to compute `val_baseline_mae` and `improvement_over_baseline` in logs and plots. If you skip it, training still runs but those metrics are omitted.
- **Synthetic**: Same idea. Use `--input data/processed/tiles/synthetic_parenthesis_512/filtered_tiles.json` and `--targets-dir data/processed/tiles/synthetic_parenthesis_512/targets`. Use `--lobe-threshold 1.0` so pixels with value 20 are treated as lobe. The synthetic pipeline runs this by default unless you pass `--skip-baselines`.

---

## run_full_pipeline.sh (generic)

Runs the full workflow in order:

1. **Data preparation** — `prepare_training_data.py` (tiles, proximity maps, filtered list)
2. **Extended training set** — `prepare_extended_training_set.py` (optional)
3. **HP tuning** — `tune_hyperparameters.py` (optional)
4. **Training** — `train_model.py`

**Usage:**

```bash
bash pipelines/run_full_pipeline.sh [--dev] [--n-trials N] [--skip-extended] [--skip-tuning]
```

| Option | Description |
|--------|-------------|
| `--dev` | Use dev data |
| `--n-trials N` | Number of Optuna trials (default: 30) |
| `--skip-extended` | Skip step 2 |
| `--skip-tuning` | Skip step 3; train with config only |

Requires Poetry and project dependencies. Set `configs/data_preparation_config.yaml` and `configs/training_config.yaml` as needed (e.g. `tile_size`, `use_background_and_augmentation`).
