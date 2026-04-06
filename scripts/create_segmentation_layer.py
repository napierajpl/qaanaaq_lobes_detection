"""
Create a segmentation layer from any raster (independent of dataset).
Output is a separate raster (same grid): segment IDs as hint for CNN boundaries.
Can be run on normal imagery or on imagery with parenthesis. Optional 2 scales.
For large rasters, uses block processing to avoid loading full image.
"""

import argparse
from pathlib import Path

from src.data_processing.segmentation_layer import create_segmentation_layer
from src.utils.path_utils import get_project_root, resolve_path


def main() -> None:
    project_root = get_project_root(Path(__file__))
    default_input = project_root / "data/raw/raster/imagery/qaanaaq_rgb_0_2m.tif"
    default_output = project_root / "data/processed/raster/imagery_segmentation_layer.tif"
    default_boundary = project_root / "data/raw/vector/research_boundary.shp"

    parser = argparse.ArgumentParser(
        description="Create segmentation layer from raster (separate layer for CNN hints).",
    )
    parser.add_argument("-i", "--input", type=Path, default=default_input)
    parser.add_argument("-o", "--output", type=Path, default=default_output)
    parser.add_argument("--scale", type=float, default=100.0, help="Felzenszwalb scale (higher = larger segments)")
    parser.add_argument("--scale2", type=float, default=None, help="Optional second scale for second band")
    parser.add_argument("--sigma", type=float, default=0.8, help="Felzenszwalb sigma")
    parser.add_argument("--block-size", type=int, default=4096, help="Block size when raster is larger")
    parser.add_argument("-b", "--boundary", type=Path, default=default_boundary, help="Limit segmentation to inside this vector (nodata outside)")
    args = parser.parse_args()

    input_path = resolve_path(args.input, project_root)
    output_path = resolve_path(args.output, project_root)
    boundary_path = resolve_path(args.boundary, project_root) if args.boundary else None

    if not input_path.exists():
        raise FileNotFoundError(f"Input raster not found: {input_path}")

    create_segmentation_layer(
        input_path,
        output_path,
        scale=args.scale,
        scale2=args.scale2,
        sigma=args.sigma,
        block_size=args.block_size,
        boundary_path=boundary_path,
    )
    print(f"Wrote segmentation layer to {output_path}")


if __name__ == "__main__":
    main()
