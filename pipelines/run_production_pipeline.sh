#!/usr/bin/env bash
#
# Production pipeline: data preparation (512) → extended set → optional tuning → training.
# Run from project root: bash pipelines/run_production_pipeline.sh [OPTIONS]
#
# Options:
#   --n-trials N       Number of Optuna trials (default: 30)
#   --skip-extended    Skip prepare_extended_training_set
#   --skip-tuning      Skip HP tuning; train with current config only
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

N_TRIALS=30
SKIP_EXTENDED=""
SKIP_TUNING=""

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
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--n-trials N] [--skip-extended] [--skip-tuning]" >&2
      exit 1
      ;;
  esac
done

echo "=== Production pipeline (512x512) ==="
echo ""

echo "=== 1/4 Data preparation (production) ==="
poetry run python scripts/prepare_training_data.py --tile-size 512

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
  poetry run python scripts/tune_hyperparameters.py --n-trials "$N_TRIALS" --pruning
fi

echo ""
echo "=== 4/4 Training (production) ==="
if [[ -n "$SKIP_TUNING" ]]; then
  poetry run python scripts/train_model.py
else
  poetry run python scripts/train_model.py --best-hparams
fi

echo ""
echo "=== Production pipeline finished ==="
