#!/usr/bin/env python3
"""Script to crop raster images from specified coordinates."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.raster_utils import crop_raster
from src.utils.cli_utils import BaseCLIParser
from src.utils.path_utils import get_project_root, resolve_path


def main():
    """Crop raster from specified coordinates."""
    project_root = get_project_root(__file__)

    parser = BaseCLIParser(
        description="Crop raster image from specified coordinates and dimensions",
        project_root=project_root,
    )

    default_input = project_root / "data" / "raw" / "raster" / "imagery" / "qaanaaq_rgb_0_2m.tif"

    parser.parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=default_input,
        help=f"Input raster file (default: {default_input.relative_to(project_root)})",
    )

    parser.parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output raster file (auto-generated from input name if not specified)",
    )

    parser.parser.add_argument(
        "--x",
        type=float,
        default=None,
        help="Top-left X coordinate (easting or longitude if --geo is used)",
    )

    parser.parser.add_argument(
        "--y",
        type=float,
        default=None,
        help="Top-left Y coordinate (northing or latitude if --geo is used)",
    )

    parser.parser.add_argument(
        "--lon",
        type=float,
        default=None,
        help="Top-left longitude (use with --lat, implies --geo)",
    )

    parser.parser.add_argument(
        "--lat",
        type=float,
        default=None,
        help="Top-left latitude (use with --lon, implies --geo)",
    )

    parser.parser.add_argument(
        "--geo",
        action="store_true",
        help="Interpret coordinates as geographic (lon/lat in EPSG:4326)",
    )

    parser.parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Crop width in pixels (default: 1024)",
    )

    parser.parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Crop height in pixels (default: 1024)",
    )

    parser.set_epilog("""
Examples:
  # Use geographic coordinates (lat/lon)
  python scripts/crop_raster.py -i input.tif --lon -69.267715 --lat 77.476578

  # Use projected coordinates
  python scripts/crop_raster.py -i input.tif --x -559723.06 --y -1241589.05

  # Custom dimensions
  python scripts/crop_raster.py -i input.tif --lon -69.267715 --lat 77.476578 --width 512 --height 512

  # Specify output file
  python scripts/crop_raster.py -i input.tif --lon -69.267715 --lat 77.476578 -o output.tif
    """)

    args = parser.parse_args()

    if args.lon is not None and args.lat is not None:
        use_geographic = True
        top_left_x = args.lon
        top_left_y = args.lat
        coord_type = "geographic (lon/lat)"
    elif args.geo:
        if args.x is None or args.y is None:
            parser.error("--geo requires --x and --y to be specified")
        use_geographic = True
        top_left_x = args.x
        top_left_y = args.y
        coord_type = "geographic (lon/lat)"
    else:
        if args.x is None or args.y is None:
            parser.error("Either --x/--y or --lon/--lat must be specified")
        use_geographic = False
        top_left_x = args.x
        top_left_y = args.y
        coord_type = "projected"

    input_path = resolve_path(args.input, project_root)
    output_path = resolve_path(args.output, project_root) if args.output is not None else None

    print(f"Input raster: {input_path}")
    print(f"Crop coordinates ({coord_type}): ({top_left_x}, {top_left_y})")
    print(f"Crop dimensions: {args.width} x {args.height} pixels")

    result_path = crop_raster(
        input_raster_path=input_path,
        top_left_x=top_left_x,
        top_left_y=top_left_y,
        width_pixels=args.width,
        height_pixels=args.height,
        output_raster_path=output_path,
        use_geographic=use_geographic,
    )

    print(f"Cropped raster saved: {result_path}")


if __name__ == "__main__":
    main()
