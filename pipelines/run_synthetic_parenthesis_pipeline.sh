#!/usr/bin/env bash
#
# Synthetic parenthesis pipeline (sanity-check): generate 512 dataset → optional baselines → training.
# Uses configs/training_config_synthetic_parenthesis.yaml for training.
# Run from project root: bash pipelines/run_synthetic_parenthesis_pipeline.sh [OPTIONS]
#
# Prerequisite: Dev 512 or production 512 source tiles must exist (for generation).
#   e.g. run: poetry run python scripts/prepare_training_data.py --dev --tile-size 512
#
# Options:
#   --skip-baselines     Do not run recalculate_baselines (training still works; no improvement_over_baseline in logs)
#   --max-tiles N        Max tiles to generate (default: all from source). Use a small number for a quick run.
#   --source dev|prod    Source for generation: dev = dev/train or dev/train_512, prod = train or train_512 (default: dev)
#   --tile-size 256|512  Tile size for synthetic dataset and training (default: 512).
#   --config PATH        Training config (default: configs/training_config_synthetic_parenthesis.yaml).
#   --filter-by-boundary Restrict tiles to research boundary (fewer empty tiles; needs data/raw/vector/research_boundary.shp).
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

SKIP_BASELINES=""
MAX_TILES=""
SOURCE="dev"
TILE_SIZE="512"
TRAINING_CONFIG="configs/training_config_synthetic_parenthesis.yaml"
FILTER_BY_BOUNDARY=""
BOUNDARY_DEFAULT="data/raw/vector/research_boundary.shp"

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-baselines)
      SKIP_BASELINES=1
      shift
      ;;
    --max-tiles)
      MAX_TILES="$2"
      shift 2
      ;;
    --source)
      SOURCE="$2"
      shift 2
      ;;
    --tile-size)
      TILE_SIZE="$2"
      shift 2
      ;;
    --config)
      TRAINING_CONFIG="$2"
      shift 2
      ;;
    --filter-by-boundary)
      FILTER_BY_BOUNDARY=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--skip-baselines] [--max-tiles N] [--source dev|prod] [--tile-size 256|512] [--config PATH] [--filter-by-boundary]" >&2
      exit 1
      ;;
  esac
done

if [[ "$TILE_SIZE" == "256" ]]; then
  if [[ "$SOURCE" == "prod" ]]; then
    SRC_FILTERED="data/processed/tiles/train/filtered_tiles.json"
    SRC_FEATURES="data/processed/tiles/train/features"
  else
    SRC_FILTERED="data/processed/tiles/dev/train/filtered_tiles.json"
    SRC_FEATURES="data/processed/tiles/dev/train/features"
  fi
  OUT_DIR="data/processed/tiles/synthetic_parenthesis_256"
else
  if [[ "$SOURCE" == "prod" ]]; then
    SRC_FILTERED="data/processed/tiles/train_512/filtered_tiles.json"
    SRC_FEATURES="data/processed/tiles/train_512/features"
  else
    SRC_FILTERED="data/processed/tiles/dev/train_512/filtered_tiles.json"
    SRC_FEATURES="data/processed/tiles/dev/train_512/features"
  fi
  OUT_DIR="data/processed/tiles/synthetic_parenthesis_512"
fi

FILTERED="$OUT_DIR/filtered_tiles.json"
TARGETS_DIR="$OUT_DIR/targets"

echo "=== Synthetic parenthesis pipeline (${TILE_SIZE}x${TILE_SIZE}) ==="
echo ""

echo "=== 1/3 Generate synthetic dataset ==="
GEN_OPTS="--source-filtered-tiles $SRC_FILTERED --source-features-dir $SRC_FEATURES --output-dir $OUT_DIR --tile-size $TILE_SIZE"
if [[ -n "$MAX_TILES" ]]; then
  GEN_OPTS="$GEN_OPTS --max-tiles $MAX_TILES"
fi
poetry run python scripts/generate_synthetic_parenthesis_dataset.py $GEN_OPTS

if [[ -n "$FILTER_BY_BOUNDARY" ]]; then
  echo ""
  echo "=== 1b/3 Filter tiles by research boundary ==="
  if [[ -f "$BOUNDARY_DEFAULT" ]] && [[ -f "$FILTERED" ]] && [[ -d "$OUT_DIR/features" ]]; then
    poetry run python scripts/filter_tiles_by_boundary.py \
      --filtered-tiles "$FILTERED" \
      -b "$BOUNDARY_DEFAULT" \
      -o "$FILTERED" \
      --features-dir "$OUT_DIR/features"
  else
    echo "Skipped: need $BOUNDARY_DEFAULT, $FILTERED, and $OUT_DIR/features (use full-raster generator for correct bounds)."
  fi
fi

echo ""
echo "=== 2/3 Recalculate baselines (optional) ==="
if [[ -n "$SKIP_BASELINES" ]]; then
  echo "Skipped (--skip-baselines). Training will not log improvement_over_baseline."
else
  if [[ -f "$FILTERED" ]] && [[ -d "$TARGETS_DIR" ]]; then
    poetry run python scripts/recalculate_baselines.py --input "$FILTERED" --targets-dir "$TARGETS_DIR" --lobe-threshold 1.0
  else
    echo "Skipped (filtered_tiles or targets not found)"
  fi
fi

echo ""
echo "=== 3/3 Training (synthetic_parenthesis mode, tile_size=$TILE_SIZE, config=$TRAINING_CONFIG) ==="
TRAIN_OPTS="--mode synthetic_parenthesis --config $TRAINING_CONFIG --tile-size $TILE_SIZE"
if [[ -n "$MAX_TILES" ]]; then
  TRAIN_OPTS="$TRAIN_OPTS --max-tiles $MAX_TILES"
fi
poetry run python scripts/train_model.py $TRAIN_OPTS

echo ""
echo "=== Synthetic parenthesis pipeline finished ==="
