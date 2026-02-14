#!/usr/bin/env python3
"""Script to stack multiple rasters into a single multi-band raster."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.raster_utils import stack_rasters
from src.utils.cli_utils import BaseCLIParser
from src.utils.path_utils import get_project_root, resolve_path


def main():
    """Stack multiple rasters into a single file."""
    project_root = get_project_root(Path(__file__))

    parser = BaseCLIParser(
        description="Stack multiple raster files into a single multi-band raster",
        project_root=project_root,
    )

    default_output = project_root / "data" / "processed" / "raster" / "features_combined.tif"

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
        help=f"Output stacked raster (default: {default_output.relative_to(project_root)})",
    )

    parser.set_epilog("""
Examples:
  # Stack RGB + DEM + Slope
  python scripts/stack_rasters.py -i rgb.tif dem.tif slope.tif -o combined.tif

  # Use default output
  python scripts/stack_rasters.py -i rgb.tif dem.tif slope.tif
    """)

    args = parser.parse_args()

    input_paths = [resolve_path(path, project_root) for path in args.inputs]
    output_path = resolve_path(args.output, project_root)

    print(f"Stacking {len(input_paths)} rasters:")
    for i, path in enumerate(input_paths, 1):
        print(f"  {i}. {path}")
    print(f"\nOutput: {output_path}")

    result_path = stack_rasters(input_paths, output_path)

    print(f"\nStacked raster saved: {result_path}")


if __name__ == "__main__":
    main()
