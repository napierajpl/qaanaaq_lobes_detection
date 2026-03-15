#!/usr/bin/env python3
"""Generate shapefile from tile registry for QGIS visualization."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.map_overlays.tile_registry import TileRegistry
from src.map_overlays.shapefile_generator import generate_tile_index_shapefile
from src.map_overlays.tile_index_utils import resolve_background_train_ids
from src.utils.path_utils import get_project_root, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    project_root = get_project_root(Path(__file__))

    parser = argparse.ArgumentParser(
        description="Generate shapefile from tile registry for QGIS. "
        "Use --tile-size 512 for 512x512 tiles (registry must exist in train_512/). "
        "Use --filtered-tiles + --features-dir + --targets-dir to add train_usage (incl. background_train)."
    )
    parser.add_argument("--registry", type=Path, default=None, help="Path to tile_registry.json (default: from --tile-size)")
    parser.add_argument("--tile-size", type=int, choices=[256, 512], default=256, help="Tile size: 256 or 512 (default: 256)")
    parser.add_argument("--output", type=Path, default=None, help="Output shapefile path (default: same dir as registry, tile_index.shp)")
    parser.add_argument("--valid-only", action="store_true", help="Only include valid tiles (filtered tiles)")
    parser.add_argument("--extended-tiles", type=Path, default=None, help="Path to extended_training_tiles.json (enables train_usage)")
    parser.add_argument("--filtered-tiles", type=Path, default=None, help="Path to filtered_tiles.json (use with --features-dir and --targets-dir)")
    parser.add_argument("--features-dir", type=Path, default=None, help="Features tile dir (for background_train_ids when --filtered-tiles is set)")
    parser.add_argument("--targets-dir", type=Path, default=None, help="Targets tile dir (for background_train_ids when --filtered-tiles is set)")
    parser.add_argument("--white-threshold", type=float, default=0.95, help="Exclude background candidates with >= this fraction white pixels")
    parser.add_argument("--n-background-and-augmented", type=int, default=None, help="Cap for background tiles to add (default: same as train size)")
    parser.add_argument("--train-split", type=float, default=0.7, help="Train split fraction for computing n_add")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling background tiles")
    args = parser.parse_args()

    if args.registry is not None:
        registry_path = resolve_path(args.registry, project_root)
    else:
        subdir = "train_512" if args.tile_size == 512 else "train"
        registry_path = project_root / "data/processed/tiles" / subdir / "tile_registry.json"

    if args.output:
        output_path = resolve_path(args.output, project_root)
    else:
        output_path = registry_path.parent / "tile_index.shp"

    if not registry_path.exists():
        logger.error(f"Registry file not found: {registry_path}")
        sys.exit(1)

    extended_path = resolve_path(args.extended_tiles, project_root) if args.extended_tiles else None
    filtered_path = resolve_path(args.filtered_tiles, project_root) if args.filtered_tiles else None
    features_dir = resolve_path(args.features_dir, project_root) if args.features_dir else None
    targets_dir = resolve_path(args.targets_dir, project_root) if args.targets_dir else None
    if args.filtered_tiles is not None and (features_dir is None or targets_dir is None):
        parent = registry_path.parent
        features_dir = features_dir or parent / "features"
        targets_dir = targets_dir or parent / "targets"

    background_train_ids = resolve_background_train_ids(
        extended_path=extended_path,
        filtered_path=filtered_path,
        features_dir=features_dir,
        targets_dir=targets_dir,
        train_split=args.train_split,
        seed=args.seed,
        white_threshold=args.white_threshold,
        n_background_and_augmented=args.n_background_and_augmented,
    )

    logger.info(f"Loading tile registry: {registry_path}")
    registry = TileRegistry(registry_path)

    generate_tile_index_shapefile(
        registry=registry,
        output_path=output_path,
        include_all_tiles=not args.valid_only,
        background_train_ids=background_train_ids,
    )

    logger.info(f"Shapefile generated successfully: {output_path}")


if __name__ == "__main__":
    main()
