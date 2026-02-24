"""
Create slope-stripes channel raster: one band (texture strength × slope alignment) from RGB + DEM.
Supports structure-tensor or Gabor method. Process whole area first; output same grid as reference (RGB).
For large rasters, uses block processing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio
from rasterio.windows import Window

from src.preprocessing.texture_hints import (
    compute_slope_stripes_channel,
    compute_gabor_slope_stripes_channel,
)
from src.utils.path_utils import get_project_root, resolve_path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, *args, **kwargs):
        return it


def _compute_block_structure_tensor(
    rgb: np.ndarray,
    dem: np.ndarray,
    sigma_smooth: float,
    sigma_structure: float,
) -> np.ndarray:
    return compute_slope_stripes_channel(
        rgb, dem, sigma_smooth=sigma_smooth, sigma_structure=sigma_structure
    )


def _compute_block_gabor(
    rgb: np.ndarray,
    dem: np.ndarray,
    frequency: float,
    sigma: float,
    n_orientations: int = 16,
) -> np.ndarray:
    return compute_gabor_slope_stripes_channel(
        rgb, dem, frequency=frequency, sigma=sigma, n_orientations=n_orientations
    )


def create_slope_stripes_channel_raster(
    rgb_path: Path,
    dem_path: Path,
    output_path: Path,
    method: str = "gabor",
    sigma_smooth: float = 1.5,
    sigma_structure: float = 2.0,
    gabor_frequency: float = 0.15,
    gabor_sigma: float = 5.0,
    gabor_n_orientations: int = 16,
    block_size: int = 1024,
) -> None:
    """
    Read RGB and DEM (same grid), compute slope-stripes per block, write 1 band.
    method: "structure_tensor" or "gabor". Output: float32 [0, 1], same grid/CRS as rgb_path.
    """
    rgb_path = Path(rgb_path)
    dem_path = Path(dem_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def compute_block(rgb: np.ndarray, dem: np.ndarray) -> np.ndarray:
        if method == "structure_tensor":
            return _compute_block_structure_tensor(rgb, dem, sigma_smooth, sigma_structure)
        if method == "gabor":
            return _compute_block_gabor(
                rgb, dem, gabor_frequency, gabor_sigma, gabor_n_orientations
            )
        raise ValueError(f"Unknown method: {method}")

    with rasterio.open(rgb_path) as rgb_src, rasterio.open(dem_path) as dem_src:
        if (rgb_src.height, rgb_src.width) != (dem_src.height, dem_src.width):
            raise ValueError(
                f"RGB and DEM size mismatch: RGB {rgb_src.height}x{rgb_src.width}, "
                f"DEM {dem_src.height}x{dem_src.width}. Resample DEM to match RGB first."
            )
        H, W = rgb_src.height, rgb_src.width
        profile = rgb_src.profile.copy()
        profile.update(count=1, dtype=np.float32, nodata=None)

        if H <= block_size and W <= block_size:
            rgb_data = rgb_src.read([1, 2, 3])
            dem_data = dem_src.read(1)
            out_band = compute_block(rgb_data, dem_data)
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(out_band.astype(np.float32), 1)
            return

        windows = [
            Window(c, r, min(block_size, W - c), min(block_size, H - r))
            for r in range(0, H, block_size)
            for c in range(0, W, block_size)
        ]
        with rasterio.open(output_path, "w", **profile) as dst:
            for window in tqdm(windows, desc="Slope-stripes blocks"):
                rgb_block = rgb_src.read([1, 2, 3], window=window)
                dem_block = dem_src.read(1, window=window)
                out_block = compute_block(rgb_block, dem_block)
                dst.write(out_block.astype(np.float32), 1, window=window)


def main() -> None:
    import argparse

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
    parser.add_argument(
        "--method",
        choices=("structure_tensor", "gabor"),
        default="gabor",
        help="Method: structure_tensor or gabor (default: gabor)",
    )
    parser.add_argument("--sigma-smooth", type=float, default=1.5, help="Structure tensor smoothing")
    parser.add_argument("--sigma-structure", type=float, default=2.0, help="Structure tensor integration scale")
    parser.add_argument("--gabor-frequency", type=float, default=0.15, help="Gabor frequency (default 0.15)")
    parser.add_argument("--gabor-sigma", type=float, default=5.0, help="Gabor sigma (default 5.0)")
    parser.add_argument("--gabor-n-orientations", type=int, default=16, help="Gabor orientations")
    parser.add_argument("--block-size", type=int, default=1024, help="Block size for large rasters (default 1024)")
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
        block_size=args.block_size,
    )
    print(f"Wrote slope-stripes channel ({args.method}) to {output_path}")


if __name__ == "__main__":
    main()
