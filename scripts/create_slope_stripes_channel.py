"""
Create slope-stripes channel raster: one band (texture strength × slope alignment) from RGB + DEM.
Supports structure-tensor or Gabor method. Process whole area first; output same grid as reference (RGB).
For large rasters, uses block processing.
"""

import argparse
from pathlib import Path

from src.preprocessing.slope_stripes_raster import create_slope_stripes_channel_raster
from src.utils.path_utils import get_project_root, resolve_path


def main() -> None:
    project_root = get_project_root(Path(__file__))
    default_rgb = project_root / "data/raw/raster/imagery/qaanaaq_rgb_0_2m.tif"
    default_dem = project_root / "data/processed/raster/dem_from_arcticDEM_resampled.tif"
    default_out = project_root / "data/processed/raster/slope_stripes_gabor_channel.tif"

    parser = argparse.ArgumentParser(
        description="Create slope-stripes channel raster (1 band, 0–1) from RGB + DEM.",
    )
    parser.add_argument("-i", "--rgb", type=Path, default=default_rgb, help="RGB raster (reference grid)")
    parser.add_argument("-d", "--dem", type=Path, default=default_dem, help="DEM raster (same grid as RGB)")
    parser.add_argument("-o", "--output", type=Path, default=default_out, help="Output GeoTIFF")
    parser.add_argument("--method", choices=("structure_tensor", "gabor"), default="gabor", help="Method (default: gabor)")
    parser.add_argument("--sigma-smooth", type=float, default=1.5, help="Structure tensor smoothing")
    parser.add_argument("--sigma-structure", type=float, default=2.0, help="Structure tensor integration scale")
    parser.add_argument("--gabor-frequency", type=float, default=0.15, help="Gabor frequency")
    parser.add_argument("--gabor-sigma", type=float, default=5.0, help="Gabor sigma")
    parser.add_argument("--gabor-n-orientations", type=int, default=16, help="Gabor orientations")
    parser.add_argument("--alignment-power", type=float, default=1.0, help="Orientation alignment power (default 1.0)")
    parser.add_argument("--block-size", type=int, default=2048, help="Block size for large rasters")
    args = parser.parse_args()

    rgb_path = resolve_path(args.rgb, project_root)
    dem_path = resolve_path(args.dem, project_root)
    output_path = resolve_path(args.output, project_root)

    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB raster not found: {rgb_path}")
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM raster not found: {dem_path}")

    create_slope_stripes_channel_raster(
        rgb_path,
        dem_path,
        output_path,
        method=args.method,
        sigma_smooth=args.sigma_smooth,
        sigma_structure=args.sigma_structure,
        gabor_frequency=args.gabor_frequency,
        gabor_sigma=args.gabor_sigma,
        gabor_n_orientations=args.gabor_n_orientations,
        alignment_power=args.alignment_power,
        block_size=args.block_size,
    )
    print(f"Wrote slope-stripes channel ({args.method}) to {output_path}")


if __name__ == "__main__":
    main()
