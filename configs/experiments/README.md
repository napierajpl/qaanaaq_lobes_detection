# Experiment configs

Each YAML file contains **overrides** applied on top of `training_config.yaml`.
Only specify keys you want to change; everything else inherits from the base config.

## Usage

Run a single experiment:
```bash
poetry run python scripts/run_experiment_sequence.py --experiments exp_01_baseline.yaml
```

Run the full sequence (sorted `exp_*.yaml` order: 00 → 04):
```bash
poetry run python scripts/run_experiment_sequence.py --all
```

## Naming convention

`exp_XX_short_name.yaml` — **lower XX = higher priority** (listed first in `--all`).
`exp_00_all_tiles.yaml` is all illumination (sun + shadow); `exp_01`–`exp_02` are sun-only probes.

Default channel stack for production BCE runs is **RGB + segmentation + slope_stripes** (no raw DEM/slope);
experiments inherit this unless they override `data.*`.

**`exp_03_bce_pos_weight`** and **`exp_04_unfreeze_encoder`** set `init_weights_from: data/models/production/best_model.pt`
so a full `--all` run chains checkpoints (after each run, `best_model.pt` is updated). For a standalone run, copy
your source checkpoint to that path or change `init_weights_from` in the YAML.
