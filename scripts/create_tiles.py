#!/usr/bin/env python3
"""Script to create overlapping tiles from raster images."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.tiling import Tiler
from src.utils.cli_utils import BaseCLIParser
from src.utils.path_utils import get_project_root, resolve_path


def main():
    """Create overlapping tiles from raster."""
    project_root = get_project_root(__file__)

    parser = BaseCLIParser(
        description="Create overlapping tiles from raster images for CNN training",
        project_root=project_root,
    )

    parser.parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Input raster file to tile",
    )

    parser.parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for tiles",
    )

    parser.parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Tile size in pixels (default: 256)",
    )

    parser.parser.add_argument(
        "--overlap",
        type=float,
        default=0.3,
        help="Overlap ratio 0.0-1.0 (default: 0.3 = 30%%)",
    )

    parser.parser.add_argument(
        "--base-name",
        type=str,
        default=None,
        help="Base name for tiles (uses input filename if not specified)",
    )

    parser.parser.add_argument(
        "--no-organize",
        action="store_true",
        help="Don't organize tiles in subfolder by source filename",
    )

    parser.set_epilog("""
Examples:
  # Create 256x256 tiles with 30% overlap
  python scripts/create_tiles.py -i input.tif -o tiles/

  # Custom tile size and overlap
  python scripts/create_tiles.py -i input.tif -o tiles/ --tile-size 512 --overlap 0.2
    """)

    args = parser.parse_args()

    input_path = resolve_path(args.input, project_root)
    output_dir = resolve_path(args.output_dir, project_root)

    print(f"Input raster: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Tile size: {args.tile_size}×{args.tile_size} pixels")
    print(f"Overlap: {args.overlap * 100:.0f}% (stride: {int(args.tile_size * (1 - args.overlap))} pixels)")

    tiler = Tiler(tile_size=args.tile_size, overlap=args.overlap)
    output_paths = tiler.tile_raster(
        input_path, output_dir, args.base_name, organize_by_source=not args.no_organize
    )

    print(f"\nCreated {len(output_paths)} tiles")
    if not args.no_organize:
        print(f"Tiles organized in: {output_dir / Path(input_path).stem}")
    else:
        print(f"Tiles saved in: {output_dir}")


if __name__ == "__main__":
    main()
