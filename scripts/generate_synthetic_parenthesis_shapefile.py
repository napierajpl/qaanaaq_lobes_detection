#!/usr/bin/env python3
"""
Generate a shapefile of synthetic parenthesis tile footprints for QGIS.

Uses the tile registry (creates it from the synthetic source raster if missing),
then calls the existing generate_tile_index_shapefile. Requires either an existing
registry or the source raster data/processed/raster/synthetic_parenthesis/synthetic_features_5band.tif.

Usage:
  poetry run python scripts/generate_synthetic_parenthesis_shapefile.py --tile-size 256
  poetry run python scripts/generate_synthetic_parenthesis_shapefile.py --tile-size 512 --output path/to/tile_index.shp
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.map_overlays.tile_registry import TileRegistry
from src.map_overlays.shapefile_generator import generate_tile_index_shapefile
from src.utils.path_utils import get_project_root, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OVERLAP = 0.3


def main():
    import argparse

    project_root = get_project_root(Path(__file__))

    parser = argparse.ArgumentParser(
        description="Generate shapefile of synthetic parenthesis tiles for QGIS."
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        choices=[256, 512],
        default=256,
        help="Tile size (256 or 512). Default: 256.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output shapefile path (default: <tile_dir>/tile_index.shp)",
    )
    parser.add_argument(
        "--tiles-dir",
        type=Path,
        default=None,
        help="Override tiles directory (default: data/processed/tiles/synthetic_parenthesis_<tile_size>)",
    )
    parser.add_argument(
        "--source-raster",
        type=Path,
        default=None,
        help="Override source raster for registry creation (default: data/processed/raster/synthetic_parenthesis/synthetic_features_5band.tif)",
    )
    args = parser.parse_args()

    if args.tiles_dir is not None:
        tiles_dir = resolve_path(args.tiles_dir, project_root)
    else:
        tiles_dir = project_root / "data/processed/tiles" / f"synthetic_parenthesis_{args.tile_size}"

    if not tiles_dir.exists():
        logger.error(f"Tiles directory not found: {tiles_dir}")
        sys.exit(1)

    filtered_path = tiles_dir / "filtered_tiles.json"
    if not filtered_path.exists():
        logger.error(f"filtered_tiles.json not found: {filtered_path}")
        sys.exit(1)

    if args.output is not None:
        output_path = resolve_path(args.output, project_root)
    else:
        output_path = tiles_dir / "tile_index.shp"

    registry_path = tiles_dir / "tile_registry.json"
    if not registry_path.exists():
        source_raster = args.source_raster
        if source_raster is None:
            source_raster = project_root / "data/processed/raster/synthetic_parenthesis/synthetic_features_5band.tif"
        else:
            source_raster = resolve_path(source_raster, project_root)
        if not source_raster.exists():
            logger.error(
                f"Registry not found and source raster not found: {source_raster}. "
                "Run generate_synthetic_parenthesis_from_raster.py first, or pass --source-raster."
            )
            sys.exit(1)
        logger.info(f"Creating tile registry from {source_raster}")
        features_dir = tiles_dir / "features"
        registry = TileRegistry(registry_path, source_raster)
        registry.migrate_from_filtered_tiles(
            filtered_tiles_path=filtered_path,
            source_raster_path=source_raster,
            features_dir=features_dir,
            train_split=0.6,
            val_split=0.2,
            test_split=0.2,
            tile_size=args.tile_size,
            overlap=OVERLAP,
        )
        boundary = project_root / "data/raw/vector/research_boundary.shp"
        if boundary.exists():
            registry.add_boundary_info(boundary)
        registry.save()
        logger.info(f"Saved {registry_path}")

    registry = TileRegistry(registry_path)
    generate_tile_index_shapefile(
        registry=registry,
        output_path=output_path,
        include_all_tiles=False,
    )
    logger.info(f"Done. Add to QGIS: {output_path}")


if __name__ == "__main__":
    main()
