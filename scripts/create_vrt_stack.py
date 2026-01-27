#!/usr/bin/env python3
"""Script to create virtual raster (VRT) stack from multiple rasters."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.raster_utils import create_vrt_stack
from src.utils.cli_utils import BaseCLIParser
from src.utils.path_utils import get_project_root, resolve_path


def main():
    """Create VRT stack from multiple rasters."""
    project_root = get_project_root(__file__)

    parser = BaseCLIParser(
        description="Create virtual raster (VRT) stack from multiple raster files",
        project_root=project_root,
    )

    default_output = project_root / "data" / "processed" / "raster" / "features_combined.vrt"

    parser.parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Input raster files to stack (in order)",
    )

    parser.parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_output,
        help=f"Output VRT file (default: {default_output.relative_to(project_root)})",
    )

    parser.set_epilog("""
Examples:
  # Create VRT stack from RGB + DEM + Slope
  python scripts/create_vrt_stack.py -i rgb.tif dem.tif slope.tif -o combined.vrt

  # Use default output
  python scripts/create_vrt_stack.py -i rgb.tif dem.tif slope.tif
    """)

    args = parser.parse_args()

    input_paths = [resolve_path(path, project_root) for path in args.inputs]
    output_path = resolve_path(args.output, project_root)

    print(f"Creating VRT stack from {len(input_paths)} rasters:")
    for i, path in enumerate(input_paths, 1):
        print(f"  {i}. {path}")
    print(f"\nOutput VRT: {output_path}")
    print("(VRT files are virtual - no data is copied, saves disk space)")

    result_path = create_vrt_stack(input_paths, output_path)

    print(f"\nVRT stack created: {result_path}")


if __name__ == "__main__":
    main()
