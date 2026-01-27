#!/usr/bin/env python3
"""Script to resample a raster to match another raster's transform and dimensions."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.raster_utils import resample_raster_to_match
from src.utils.cli_utils import BaseCLIParser
from src.utils.path_utils import get_project_root, resolve_path
import rasterio.warp


def main():
    """Resample raster to match reference raster."""
    project_root = get_project_root(__file__)

    parser = BaseCLIParser(
        description="Resample a raster to match another raster's transform, CRS, and dimensions",
        project_root=project_root,
    )

    parser.parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Input raster file to resample",
    )

    parser.parser.add_argument(
        "-r",
        "--reference",
        type=Path,
        required=True,
        help="Reference raster to match (transform, CRS, dimensions)",
    )

    parser.parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output resampled raster file",
    )

    parser.parser.add_argument(
        "--method",
        type=str,
        default="bilinear",
        choices=["nearest", "bilinear", "cubic", "average"],
        help="Resampling method (default: bilinear)",
    )

    parser.set_epilog("""
Examples:
  # Resample DEM to match RGB raster resolution
  python scripts/resample_raster.py -i dem.tif -r rgb.tif -o dem_resampled.tif

  # Use nearest neighbor for categorical data
  python scripts/resample_raster.py -i input.tif -r reference.tif -o output.tif --method nearest
    """)

    args = parser.parse_args()

    method_map = {
        "nearest": rasterio.warp.Resampling.nearest,
        "bilinear": rasterio.warp.Resampling.bilinear,
        "cubic": rasterio.warp.Resampling.cubic,
        "average": rasterio.warp.Resampling.average,
    }

    input_path = resolve_path(args.input, project_root)
    reference_path = resolve_path(args.reference, project_root)
    output_path = resolve_path(args.output, project_root)

    print(f"Input raster: {input_path}")
    print(f"Reference raster: {reference_path}")
    print(f"Output raster: {output_path}")
    print(f"Resampling method: {args.method}")

    result_path = resample_raster_to_match(
        input_raster_path=input_path,
        reference_raster_path=reference_path,
        output_raster_path=output_path,
        resampling_method=method_map[args.method],
    )

    print(f"Resampled raster saved: {result_path}")


if __name__ == "__main__":
    main()
