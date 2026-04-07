# Experiment configs

Each YAML file contains **overrides** applied on top of `training_config.yaml`.
Only specify keys you want to change; everything else inherits from the base config.

## Usage

Run a single experiment:
```bash
poetry run python scripts/run_experiment_sequence.py --experiments exp_00_baseline.yaml
```

Run a batch (sequentially):
```bash
poetry run python scripts/run_experiment_sequence.py --experiments exp_00_baseline.yaml exp_01_augmentation.yaml exp_02_all_tiles.yaml
```

## Naming convention

`exp_XX_short_name.yaml` — XX is the order, short_name describes the hypothesis.
