#!/usr/bin/env bash
#
# Dev pipeline: data preparation (256) → extended set → optional tuning → training.
# Run from project root: bash pipelines/run_dev_pipeline.sh [OPTIONS]
#
# Options:
#   --n-trials N       Number of Optuna trials (default: 30)
#   --skip-extended    Skip prepare_extended_training_set
#   --skip-tuning      Skip HP tuning; train with current config only
#   --tile-size 256|512  Tile size for data prep (default: 256). Training uses config.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

N_TRIALS=30
SKIP_EXTENDED=""
SKIP_TUNING=""
TILE_SIZE=256

while [[ $# -gt 0 ]]; do
  case $1 in
    --n-trials)
      N_TRIALS="$2"
      shift 2
      ;;
    --skip-extended)
      SKIP_EXTENDED=1
      shift
      ;;
    --skip-tuning)
      SKIP_TUNING=1
      shift
      ;;
    --tile-size)
      TILE_SIZE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--n-trials N] [--skip-extended] [--skip-tuning] [--tile-size 256|512]" >&2
      exit 1
      ;;
  esac
done

echo "=== Dev pipeline (tile size ${TILE_SIZE}) ==="
echo ""

echo "=== 1/4 Data preparation (dev) ==="
poetry run python scripts/prepare_training_data.py --dev --tile-size "$TILE_SIZE"

echo ""
echo "=== 2/4 Extended training set ==="
if [[ -n "$SKIP_EXTENDED" ]]; then
  echo "Skipped (--skip-extended)"
else
  poetry run python scripts/prepare_extended_training_set.py --config configs/data_preparation_config.yaml
fi

echo ""
echo "=== 3/4 Hyperparameter tuning ==="
if [[ -n "$SKIP_TUNING" ]]; then
  echo "Skipped (--skip-tuning)"
else
  poetry run python scripts/tune_hyperparameters.py --n-trials "$N_TRIALS" --dev --pruning
fi

echo ""
echo "=== 4/4 Training (dev) ==="
if [[ -n "$SKIP_TUNING" ]]; then
  poetry run python scripts/train_model.py --dev --tile-size "$TILE_SIZE"
else
  poetry run python scripts/train_model.py --dev --tile-size "$TILE_SIZE" --best-hparams
fi

echo ""
echo "=== Dev pipeline finished ==="
