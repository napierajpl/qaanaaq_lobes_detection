"""
Tile the full segmentation raster into synthetic_parenthesis_256/segmentation and
synthetic_parenthesis_512/segmentation so tile indices match features/targets.
Use after create_segmentation_for_synthetic_parenthesis.py (full raster only).
"""
from pathlib import Path

from src.data_processing.tiling import Tiler
from src.utils.path_utils import get_project_root, resolve_path

OVERLAP = 0.3
TILE_SIZES = [256, 512]


def main() -> None:
    project_root = get_project_root(Path(__file__))
    default_seg_raster = project_root / "data/processed/raster/synthetic_parenthesis/segmentation_layer.tif"
    default_tiles_base = project_root / "data/processed/tiles"

    import argparse
    parser = argparse.ArgumentParser(
        description="Tile segmentation raster to synthetic_parenthesis_256 and _512.",
    )
    parser.add_argument(
        "-i", "--segmentation-raster",
        type=Path,
        default=default_seg_raster,
        help="Full segmentation GeoTIFF (same grid as synthetic_features_5band.tif).",
    )
    parser.add_argument(
        "--tiles-base",
        type=Path,
        default=default_tiles_base,
        help="Base dir containing synthetic_parenthesis_256 and synthetic_parenthesis_512.",
    )
    parser.add_argument("--tile-sizes", type=int, nargs="+", default=TILE_SIZES)
    parser.add_argument("--overlap", type=float, default=OVERLAP)
    args = parser.parse_args()

    seg_raster_path = resolve_path(args.segmentation_raster, project_root)
    tiles_base = resolve_path(args.tiles_base, project_root)

    if not seg_raster_path.exists():
        raise FileNotFoundError(f"Segmentation raster not found: {seg_raster_path}")

    for tile_size in args.tile_sizes:
        out_dir = tiles_base / f"synthetic_parenthesis_{tile_size}" / "segmentation"
        out_dir.mkdir(parents=True, exist_ok=True)
        tiler = Tiler(tile_size=tile_size, overlap=args.overlap)
        paths = tiler.tile_raster(
            seg_raster_path,
            out_dir,
            base_filename="segmentation",
            organize_by_source=False,
        )
        print(f"Wrote {len(paths)} segmentation tiles to {out_dir} (tile_size={tile_size})")


if __name__ == "__main__":
    main()
