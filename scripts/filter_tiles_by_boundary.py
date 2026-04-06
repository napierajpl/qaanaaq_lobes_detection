"""
Filter filtered_tiles.json to only tiles that intersect the research boundary.
Use the output as filtered_tiles (or replace) to limit training to the boundary.
Run create_tile_registry first if you want to use --registry for faster filtering.
"""
from pathlib import Path

from src.data_processing.boundary_tile_filter import filter_filtered_tiles_by_boundary
from src.utils.path_utils import get_project_root, resolve_path


def main() -> None:
    project_root = get_project_root(Path(__file__))
    default_boundary = project_root / "data/raw/vector/research_boundary.shp"

    import argparse
    parser = argparse.ArgumentParser(
        description="Filter tiles to those intersecting a boundary (limit training to AOI).",
    )
    parser.add_argument(
        "--filtered-tiles",
        type=Path,
        required=True,
        help="Path to filtered_tiles.json",
    )
    parser.add_argument(
        "-b", "--boundary",
        type=Path,
        default=default_boundary,
        help=f"Boundary vector (default: {default_boundary})",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output filtered_tiles.json (only tiles inside boundary)",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Tile registry JSON (for bounds). If not set, use --features-dir.",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=None,
        help="Features tile directory (to read bounds from GeoTIFFs when no registry)",
    )
    args = parser.parse_args()

    filtered_path = resolve_path(args.filtered_tiles, project_root)
    boundary_path = resolve_path(args.boundary, project_root)
    output_path = resolve_path(args.output, project_root)
    registry_path = resolve_path(args.registry, project_root) if args.registry else None
    features_dir = resolve_path(args.features_dir, project_root) if args.features_dir else None

    if not filtered_path.exists():
        raise FileNotFoundError(f"Filtered tiles not found: {filtered_path}")
    if not boundary_path.exists():
        raise FileNotFoundError(f"Boundary not found: {boundary_path}")

    n = filter_filtered_tiles_by_boundary(
        filtered_path,
        boundary_path,
        output_path,
        features_dir=features_dir,
        registry_path=registry_path,
    )
    print(f"Wrote {n} tiles (inside boundary) to {output_path}")


if __name__ == "__main__":
    main()
