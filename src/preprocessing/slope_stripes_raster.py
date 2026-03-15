"""Create slope-stripes channel raster (1 band) from RGB + DEM."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from src.preprocessing.texture_hints import (
    compute_slope_stripes_channel,
    compute_gabor_slope_stripes_channel,
)

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
    alignment_power: float = 1.0,
) -> np.ndarray:
    return compute_slope_stripes_channel(
        rgb, dem, sigma_smooth=sigma_smooth, sigma_structure=sigma_structure,
        alignment_power=alignment_power,
    )


def _compute_block_gabor(
    rgb: np.ndarray,
    dem: np.ndarray,
    frequency: float,
    sigma: float,
    n_orientations: int = 16,
    alignment_power: float = 1.0,
) -> np.ndarray:
    return compute_gabor_slope_stripes_channel(
        rgb, dem, frequency=frequency, sigma=sigma, n_orientations=n_orientations,
        alignment_power=alignment_power,
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
    alignment_power: float = 1.0,
    block_size: int = 2048,
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
            return _compute_block_structure_tensor(
                rgb, dem, sigma_smooth, sigma_structure, alignment_power
            )
        if method == "gabor":
            return _compute_block_gabor(
                rgb, dem, gabor_frequency, gabor_sigma, gabor_n_orientations, alignment_power
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
