#!/usr/bin/env python3
"""Create tile registry from filtered_tiles.json."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.map_overlays.tile_registry import TileRegistry
from src.utils.path_utils import get_project_root, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    """Create tile registry from filtered tiles."""
    import argparse

    project_root = get_project_root(Path(__file__))

    parser = argparse.ArgumentParser(description="Create tile registry from filtered_tiles.json")
    parser.add_argument(
        "--filtered-tiles",
        type=Path,
        default=project_root / "data/processed/tiles/train/filtered_tiles.json",
        help="Path to filtered_tiles.json",
    )
    parser.add_argument(
        "--source-raster",
        type=Path,
        default=project_root / "data/raw/raster/imagery/qaanaaq_rgb_0_2m.tif",
        help="Path to source raster for geographic bounds",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=project_root / "data/processed/tiles/train/features",
        help="Directory containing feature tiles",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output registry path (default: same dir as filtered_tiles.json)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Training split fraction",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Validation split fraction",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.15,
        help="Test split fraction",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        choices=[256, 512],
        default=256,
        help="Tile size in pixels (must match the tiling used to create tiles). Default: 256.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.3,
        help="Overlap ratio used when creating tiles (default: 0.3).",
    )

    args = parser.parse_args()

    # Resolve paths
    filtered_tiles_path = resolve_path(args.filtered_tiles, project_root)
    source_raster_path = resolve_path(args.source_raster, project_root)
    features_dir = resolve_path(args.features_dir, project_root)

    if args.output:
        registry_path = resolve_path(args.output, project_root)
    else:
        registry_path = filtered_tiles_path.parent / "tile_registry.json"

    # Validate inputs
    if not filtered_tiles_path.exists():
        logger.error(f"Filtered tiles file not found: {filtered_tiles_path}")
        sys.exit(1)

    if not source_raster_path.exists():
        logger.error(f"Source raster not found: {source_raster_path}")
        sys.exit(1)

    # Create registry
    logger.info(f"Creating tile registry: {registry_path}")
    registry = TileRegistry(registry_path, source_raster_path)

    # Migrate from filtered_tiles.json
    registry.migrate_from_filtered_tiles(
        filtered_tiles_path=filtered_tiles_path,
        source_raster_path=source_raster_path,
        features_dir=features_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )

    logger.info(f"Tile registry created successfully: {registry_path}")
    logger.info(f"Total tiles in registry: {len(registry.registry['tiles'])}")


if __name__ == "__main__":
    main()
