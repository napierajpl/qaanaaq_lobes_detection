#!/usr/bin/env python3
"""Script to generate proximity map from binary raster."""

from pathlib import Path

from src.data_processing.raster_utils import generate_proximity_map
from src.utils.cli_utils import BaseCLIParser
from src.utils.path_utils import get_project_root, resolve_path


def main():
    """Generate proximity map from binary raster."""
    project_root = get_project_root(Path(__file__))

    parser = BaseCLIParser(
        description="Generate proximity map from binary raster using distance transform",
        project_root=project_root,
    )

    default_input = project_root / "data" / "processed" / "raster" / "rasterized_lobes_raw_by_code.tif"
    default_output = project_root / "data" / "processed" / "raster" / "proximity_map.tif"

    parser.add_input_output_args(
        default_input=default_input,
        default_output=default_output,
    )

    parser.parser.add_argument(
        "--max-value",
        type=int,
        default=20,
        help="Maximum proximity value for lobe pixels (default: 20)",
    )

    parser.parser.add_argument(
        "--max-distance",
        type=int,
        default=20,
        help="Maximum distance in pixels to consider (default: 20)",
    )

    parser.parser.add_argument(
        "--lobe-value",
        type=int,
        default=1,
        help="Value representing lobes in input raster (default: 1)",
    )

    parser.set_epilog("""
Examples:
  # Use default paths
  python scripts/generate_proximity_map.py

  # Specify custom input and output
  python scripts/generate_proximity_map.py -i input.tif -o output.tif

  # Custom proximity parameters
  python scripts/generate_proximity_map.py -i input.tif -o output.tif --max-value 15 --max-distance 15
    """)

    args = parser.parse_args()

    input_path = resolve_path(args.input, project_root)
    output_path = resolve_path(args.output, project_root)

    print(f"Input raster: {input_path}")
    print(f"Output proximity map: {output_path}")
    print(f"Max value: {args.max_value}, Max distance: {args.max_distance}")

    generate_proximity_map(
        input_raster_path=input_path,
        output_raster_path=output_path,
        max_value=args.max_value,
        max_distance=args.max_distance,
        lobe_value=args.lobe_value,
    )

    print(f"Proximity map generated: {output_path}")


if __name__ == "__main__":
    main()
