#!/usr/bin/env bash
#
# Full pipeline: data preparation → extended training set → HP tuning → training.
# Run from project root: bash pipelines/run_full_pipeline.sh [OPTIONS]
#
# Options:
#   --dev              Use dev data (small area, faster)
#   --n-trials N       Number of Optuna trials (default: 30)
#   --skip-extended    Skip prepare_extended_training_set (use only filtered_tiles split)
#   --skip-tuning      Skip HP tuning; run training with current config only (no --best-hparams)
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

DEV=""
N_TRIALS=30
SKIP_EXTENDED=""
SKIP_TUNING=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --dev)
      DEV="--dev"
      shift
      ;;
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
      echo "Usage: $0 [--dev] [--n-trials N] [--skip-extended] [--skip-tuning]" >&2
      exit 1
      ;;
  esac
done

echo "=== 1/4 Data preparation ==="
poetry run python scripts/prepare_training_data.py $DEV

echo ""
echo "=== 2/4 Extended training set (background + augmentation) ==="
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
  poetry run python scripts/tune_hyperparameters.py --n-trials "$N_TRIALS" $DEV --pruning
fi

echo ""
echo "=== 4/4 Training ==="
if [[ -n "$SKIP_TUNING" ]]; then
  poetry run python scripts/train_model.py $DEV
else
  poetry run python scripts/train_model.py $DEV --best-hparams
fi

echo ""
echo "=== Pipeline finished ==="
