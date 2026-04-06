#!/usr/bin/env python3
"""Add inside_boundary to an existing tile_registry.json without re-running create_tile_registry."""

import sys
from pathlib import Path

from src.map_overlays.tile_registry import TileRegistry
from src.utils.path_utils import get_project_root, resolve_path


def main() -> None:
    project_root = get_project_root(Path(__file__))
    default_boundary = project_root / "data/raw/vector/research_boundary.shp"
    import argparse
    parser = argparse.ArgumentParser(
        description="Add inside_boundary to each tile in an existing tile_registry.json.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        required=True,
        help="Path to tile_registry.json",
    )
    parser.add_argument(
        "-b", "--boundary",
        type=Path,
        default=default_boundary,
        help=f"Boundary vector (default: {default_boundary})",
    )
    args = parser.parse_args()

    registry_path = resolve_path(args.registry, project_root)
    boundary_path = resolve_path(args.boundary, project_root)
    if not registry_path.exists():
        print(f"Error: registry not found: {registry_path}")
        sys.exit(1)
    if not boundary_path.exists():
        print(f"Error: boundary not found: {boundary_path}")
        sys.exit(1)

    registry = TileRegistry(registry_path, source_raster_path=None)
    registry.add_boundary_info(boundary_path)
    registry.save()
    print(f"Updated {registry_path} with inside_boundary from {boundary_path.name}")


if __name__ == "__main__":
    main()
