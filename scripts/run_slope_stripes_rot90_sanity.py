"""
Sanity check: run slope-stripes with RGB rotated 90 degrees (DEM unchanged).
If stripes in the scene follow slope, rotated RGB should give very low values
(stripe direction in image will be perpendicular to slope orientation).
Runs both: Structure tensor (smooth=1.5, structure=3.0) and Gabor (freq=0.15, sigma=5.0).
Output: structure_tensor_rot90_sanity.png, gabor_rot90_sanity.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio

from src.preprocessing.texture_hints import (
    compute_slope_stripes_channel,
    compute_gabor_slope_stripes_channel,
    aspect_from_dem,
)
from src.preprocessing.slope_stripes_plots import make_six_panel
from src.utils.path_utils import get_project_root, resolve_path


def load_dev_rasters(project_root: Path):
    rgb_path = resolve_path(Path("data/processed/raster/dev/qaanaaq_rgb_0_2m_cropped1024x1024.tif"), project_root)
    dem_path = resolve_path(Path("data/processed/raster/dev/dem_from_arcticDEM_cropped1024x1024_resampled.tif"), project_root)
    slope_path = resolve_path(Path("data/processed/raster/dev/slope_from_dem_cropped1024x1024_resampled.tif"), project_root)
    if not rgb_path.exists() or not dem_path.exists() or not slope_path.exists():
        return None, None, None, None, None, None, None
    with rasterio.open(rgb_path) as src:
        rgb = src.read([1, 2, 3])
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
    with rasterio.open(slope_path) as src:
        slope = src.read(1)
    H, W = dem.shape
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    dem = np.asarray(dem, dtype=np.float64)
    slope = np.asarray(slope, dtype=np.float64)
    rgb_rot = np.rot90(rgb, k=1, axes=(1, 2))
    dem_safe = np.nan_to_num(dem, nan=0.0, posinf=0.0, neginf=0.0)
    slope_safe = np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0)
    aspect = aspect_from_dem(dem)
    aspect_display = np.clip((aspect.astype(np.float64) + np.pi) / (2.0 * np.pi), 0.0, 1.0)
    return rgb_rot, dem, dem_safe, slope_safe, aspect_display, H, W


def main():
    import matplotlib.pyplot as plt

    project_root = get_project_root(Path(__file__))
    out_dir = project_root / "data/processed/raster/dev"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_dev_rasters(project_root)
    if data[0] is None:
        print("Missing dev rasters. Run prepare_training_data --dev first.")
        return 1

    rgb_rot, dem, dem_safe, slope_safe, aspect_display, H, W = data

    print("Sanity check: RGB rotated 90°, DEM unchanged. Expect low slope-stripes values.")
    print()

    stripe_st = compute_slope_stripes_channel(rgb_rot, dem, sigma_smooth=1.5, sigma_structure=3.0)
    stripe_st = np.clip(stripe_st.astype(np.float64), 0.0, 1.0)
    mean_st = float(np.mean(stripe_st))
    print(f"Structure tensor (smooth=1.5, structure=3.0):  mean = {mean_st:.4f}")

    fig_st = make_six_panel(
        rgb_rot, stripe_st, dem_safe, slope_safe, aspect_display,
        "Structure tensor slope-stripes  (smooth=1.5  structure=3.0)",
        "Structure tensor  |  RGB rotated 90°  (expect low values)",
        H, W, rgb_title="RGB rotated 90 deg (1024x1024)", dem_title="DEM (unchanged)",
    )
    out_st = out_dir / "structure_tensor_rot90_sanity.png"
    fig_st.savefig(out_st, dpi=150, bbox_inches="tight")
    plt.close(fig_st)
    print(f"  -> {out_st.name}")

    stripe_gab = compute_gabor_slope_stripes_channel(rgb_rot, dem, frequency=0.15, sigma=5.0, n_orientations=16)
    stripe_gab = np.clip(stripe_gab.astype(np.float64), 0.0, 1.0)
    mean_gab = float(np.mean(stripe_gab))
    print(f"Gabor (freq=0.15, sigma=5.0):                 mean = {mean_gab:.4f}")

    fig_gab = make_six_panel(
        rgb_rot, stripe_gab, dem_safe, slope_safe, aspect_display,
        "Gabor slope-stripes  (freq=0.15  sigma=5.0)",
        "Gabor  |  RGB rotated 90°  (expect low values)",
        H, W, rgb_title="RGB rotated 90 deg (1024x1024)", dem_title="DEM (unchanged)",
    )
    out_gab = out_dir / "gabor_rot90_sanity.png"
    fig_gab.savefig(out_gab, dpi=150, bbox_inches="tight")
    plt.close(fig_gab)
    print(f"  -> {out_gab.name}")

    print()
    print("If stripes in the scene follow slope, both means should be low (rotated stripes ~ perpendicular to slope).")
    print(f"All saved in: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
