#!/usr/bin/env bash
#
# Full-raster synthetic parenthesis pipeline: generate rasters → tile 256 & 512 (features, targets, segmentation) → registries with boundary.
# Run from project root: bash pipelines/run_synthetic_from_raster_pipeline.sh [OPTIONS]
#
# Steps:
#   1) generate_synthetic_parenthesis_from_raster.py — full rasters to data/processed/raster/synthetic_parenthesis/, tile features+targets to synthetic_parenthesis_256 and _512, write filtered_tiles.json and tile_registry.json (with inside_boundary).
#   2) If segmentation_layer.tif missing: create_segmentation_for_synthetic_parenthesis.py — full raster only (no tiling).
#   3) tile_synthetic_segmentation.py — tile segmentation_layer.tif into synthetic_parenthesis_256/segmentation and synthetic_parenthesis_512/segmentation.
#
# Options (passed to generate_synthetic_parenthesis_from_raster.py):
#   --filter-by-boundary   Restrict tiles to research boundary.
#   --min-target-coverage F  Drop tiles with lobe coverage below F (e.g. 0.0001).
#   --tile-sizes 256 512    Default.
#   --overlap 0.3           Default.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

RASTER_DIR="data/processed/raster/synthetic_parenthesis"
SEG_RASTER="$RASTER_DIR/segmentation_layer.tif"

# Parse options to pass to generate script (optional)
GEN_OPTS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --filter-by-boundary)
      GEN_OPTS+=(--filter-by-boundary)
      shift
      ;;
    --min-target-coverage)
      GEN_OPTS+=(--min-target-coverage "$2")
      shift 2
      ;;
    --tile-sizes)
      GEN_OPTS+=(--tile-sizes "$2" "$3")
      shift 3
      ;;
    --overlap)
      GEN_OPTS+=(--overlap "$2")
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--filter-by-boundary] [--min-target-coverage F] [--tile-sizes 256 512] [--overlap 0.3]" >&2
      exit 1
      ;;
  esac
done

echo "=== Synthetic from-raster pipeline (256 & 512 with segmentation + registries + boundary) ==="
echo ""

echo "=== 1/3 Generate synthetic rasters and tile features + targets + registries ==="
poetry run python scripts/generate_synthetic_parenthesis_from_raster.py "${GEN_OPTS[@]}"

echo ""
echo "=== 2/3 Ensure segmentation full raster exists ==="
if [[ ! -f "$SEG_RASTER" ]]; then
  poetry run python scripts/create_segmentation_for_synthetic_parenthesis.py
else
  echo "Segmentation raster already exists: $SEG_RASTER (skipping create)"
fi

echo ""
echo "=== 3/3 Tile segmentation to 256 and 512 ==="
poetry run python scripts/tile_synthetic_segmentation.py

echo ""
echo "=== Synthetic from-raster pipeline finished ==="
echo "Tile sets: data/processed/tiles/synthetic_parenthesis_256 and synthetic_parenthesis_512 (features, targets, segmentation)."
echo "Registries: tile_registry.json in each, with inside_boundary."
