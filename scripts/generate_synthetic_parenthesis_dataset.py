"""
Generate synthetic parenthesis dataset for sanity-check training (512x512 only).

Uses existing RGB (+ DEM + Slope) tiles, draws big black "(" and ")" at random
positions/rotations, burns them into RGB, and writes target raster (20 inside shape, 0 outside).
Output layout matches normal tiles: features/, targets/, filtered_tiles.json.
Synthetic mode is 512x512 only. Train with: --mode synthetic_parenthesis
"""

import argparse
from pathlib import Path

from src.data_processing.synthetic_parenthesis import generate_synthetic_parenthesis_dataset
from src.utils.path_utils import get_project_root, resolve_path


def main() -> None:
    project_root = get_project_root(Path(__file__))
    parser = argparse.ArgumentParser(
        description="Generate synthetic parenthesis dataset from existing tiles.",
    )
    parser.add_argument(
        "--source-filtered-tiles",
        type=Path,
        default=project_root / "data/processed/tiles/dev/train_512/filtered_tiles.json",
        help="Path to source filtered_tiles.json (512 tiles)",
    )
    parser.add_argument(
        "--source-features-dir",
        type=Path,
        default=project_root / "data/processed/tiles/dev/train_512/features",
        help="Base dir for source 512x512 feature rasters",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data/processed/tiles/synthetic_parenthesis_512",
        help="Output directory (features/, targets/, filtered_tiles.json); use synthetic_parenthesis_512 for training",
    )
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size (512 only for synthetic mode)")
    parser.add_argument("--max-tiles", type=int, default=None, help="Max tiles to generate (default: all)")
    parser.add_argument("--shapes-per-tile", type=int, default=2, help="Number of ( ) shapes per tile")
    parser.add_argument("--shape-height-px", type=int, default=300, help="Height of parenthesis in px")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_filtered = resolve_path(args.source_filtered_tiles, project_root)
    source_features = resolve_path(args.source_features_dir, project_root)
    output_dir = resolve_path(args.output_dir, project_root)

    if not source_filtered.exists():
        raise FileNotFoundError(f"Source filtered tiles not found: {source_filtered}")
    if not source_features.exists():
        raise FileNotFoundError(f"Source features dir not found: {source_features}")

    generate_synthetic_parenthesis_dataset(
        source_filtered_tiles=source_filtered,
        source_features_dir=source_features,
        output_dir=output_dir,
        tile_size=args.tile_size,
        max_tiles=args.max_tiles,
        shapes_per_tile=args.shapes_per_tile,
        shape_height_px=args.shape_height_px,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
