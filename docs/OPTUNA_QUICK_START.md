# Optuna Hyperparameter Tuning - Quick Start

## Installation

First, install Optuna:
```bash
poetry install
```

## Start Hyperparameter Tuning

### Bash Command (Linux/Mac/Git Bash on Windows)

```bash
poetry run python scripts/tune_hyperparameters.py --n-trials 30 --dev --pruning
```

### Windows Command Prompt / PowerShell

```bash
poetry run python scripts/tune_hyperparameters.py --n-trials 30 --dev --pruning
```

### Using the Shell Script (Linux/Mac/Git Bash)

```bash
./START_HP_TUNING.sh
```

## What It Does

- **Runs 30 trials** (estimated ~8 hours with pruning)
- **Tunes 7 hyperparameters**:
  - Learning rate (log scale: 1e-4 to 1e-1)
  - Batch size (8, 16, 32, 64)
  - Weight decay (log scale: 1e-5 to 1e-2)
  - Focal Loss alpha (0.25 to 0.95)
  - Focal Loss gamma (1.0 to 4.0)
  - Loss function (focal, combined, weighted_smooth_l1)
  - Encoder name (resnet50, resnet152, swin_v2_base, swin_v2_tiny)
- **Uses MedianPruner** to stop underperforming trials early
- **Logs all trials** to MLflow experiment `lobe_detection_hp_tuning`
- **Saves best hyperparameters** to `configs/best_hyperparameters.yaml`

## Options

- `--n-trials N`: Number of trials (default: 30)
- `--dev`: Use dev dataset (faster, for testing)
- `--pruning`: Enable pruning (default: True)
- `--study-name NAME`: Optuna study name (default: lobe_detection_hp_tuning)
- `--config PATH`: Path to base config file

## Results

After completion:
- **Best hyperparameters**: `configs/best_hyperparameters.yaml`
- **MLflow experiment**: `lobe_detection_hp_tuning`
- **View results**: `poetry run python scripts/start_mlflow_ui.py`

## Monitor Progress

The script will show:
- Trial number and current hyperparameters
- Progress bar
- Best trial so far
- Pruned trials (stopped early)

## Stop Early

Press `Ctrl+C` to stop. The study will be saved and can be resumed later.
