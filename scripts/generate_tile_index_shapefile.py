#!/usr/bin/env python3
"""Generate shapefile from tile registry for QGIS visualization."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.map_overlays.tile_registry import TileRegistry
from src.map_overlays.shapefile_generator import generate_tile_index_shapefile
from src.utils.path_utils import get_project_root, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    """Generate shapefile from tile registry."""
    import argparse

    project_root = get_project_root(__file__)

    parser = argparse.ArgumentParser(description="Generate shapefile from tile registry")
    parser.add_argument(
        "--registry",
        type=Path,
        default=project_root / "data/processed/tiles/train/tile_registry.json",
        help="Path to tile_registry.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output shapefile path (default: same dir as registry, named tile_index.shp)",
    )
    parser.add_argument(
        "--valid-only",
        action="store_true",
        help="Only include valid tiles (filtered tiles)",
    )

    args = parser.parse_args()

    # Resolve paths
    registry_path = resolve_path(args.registry, project_root)

    if args.output:
        output_path = resolve_path(args.output, project_root)
    else:
        output_path = registry_path.parent / "tile_index.shp"

    # Validate inputs
    if not registry_path.exists():
        logger.error(f"Registry file not found: {registry_path}")
        sys.exit(1)

    # Load registry
    logger.info(f"Loading tile registry: {registry_path}")
    registry = TileRegistry(registry_path)

    # Generate shapefile
    generate_tile_index_shapefile(
        registry=registry,
        output_path=output_path,
        include_all_tiles=not args.valid_only,
    )

    logger.info(f"Shapefile generated successfully: {output_path}")


if __name__ == "__main__":
    main()
