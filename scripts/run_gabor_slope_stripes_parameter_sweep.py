"""
Run Gabor-based slope-stripes channel with 10 (frequency, sigma) combinations,
same 6-panel layout as structure-tensor sweep. Output: gabor_slope_stripes_sample_freq{X}_sigma{Y}.png
Requires: pip install scikit-image
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio
import matplotlib.pyplot as plt

from src.preprocessing.texture_hints import compute_gabor_slope_stripes_channel, aspect_from_dem
from src.preprocessing.slope_stripes_plots import make_six_panel, param_to_str
from src.utils.path_utils import get_project_root, resolve_path


def load_dev_rasters(project_root: Path):
    rgb_path = resolve_path(Path("data/processed/raster/dev/qaanaaq_rgb_0_2m_cropped1024x1024.tif"), project_root)
    dem_path = resolve_path(Path("data/processed/raster/dev/dem_from_arcticDEM_cropped1024x1024_resampled.tif"), project_root)
    slope_path = resolve_path(Path("data/processed/raster/dev/slope_from_dem_cropped1024x1024_resampled.tif"), project_root)
    if not rgb_path.exists() or not dem_path.exists() or not slope_path.exists():
        return None
    with rasterio.open(rgb_path) as src:
        rgb = src.read([1, 2, 3])
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
    with rasterio.open(slope_path) as src:
        slope = src.read(1)
    H, W = dem.shape
    if rgb.shape[1:] != (H, W) or slope.shape != (H, W):
        return None
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    dem = np.asarray(dem, dtype=np.float64)
    slope = np.asarray(slope, dtype=np.float64)
    dem_safe = np.nan_to_num(dem, nan=0.0, posinf=0.0, neginf=0.0)
    slope_safe = np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0)
    aspect = aspect_from_dem(dem)
    aspect_display = np.clip((aspect.astype(np.float64) + np.pi) / (2.0 * np.pi), 0.0, 1.0)
    return rgb, dem, dem_safe, slope_safe, aspect_display, H, W


def main():
    project_root = get_project_root(Path(__file__))
    out_dir = project_root / "data/processed/raster/dev"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_dev_rasters(project_root)
    if data is None:
        print("Missing dev rasters or shape mismatch. Run prepare_training_data --dev first.")
        return 1

    rgb, dem, dem_safe, slope_safe, aspect_display, H, W = data
    frequency, sigma = 0.15, 5.0
    alignment_powers = [0.0, 0.25, 0.5, 0.75, 1.0]

    print(f"Gabor: freq={frequency} sigma={sigma} (training values)")
    print("Sweeping alignment power (0 to 1: reduce orientation influence)...")
    print("Each figure: RGB, Gabor slope-stripes, DEM, Slope, Slope orientation, 10x10 mean (2x3 grid)")
    print()
    results = []

    for i, alignment_power in enumerate(alignment_powers, 1):
        label = f"freq{param_to_str(frequency)}sigma{param_to_str(sigma)}_align{param_to_str(alignment_power)}"
        stripe = compute_gabor_slope_stripes_channel(
            rgb, dem, frequency=frequency, sigma=sigma, n_orientations=16,
            alignment_power=alignment_power,
        )
        stripe = np.clip(stripe.astype(np.float64), 0.0, 1.0)
        mean_val = float(np.mean(stripe))
        min_val = float(np.min(stripe))
        max_val = float(np.max(stripe))
        results.append((alignment_power, mean_val, min_val, max_val, label))

        fig = make_six_panel(
            rgb, stripe, dem_safe, slope_safe, aspect_display,
            f"Gabor slope-stripes  freq={frequency}  sigma={sigma}  align_pow={alignment_power}",
            f"gabor_slope_stripes_sample_{label}.png",
            H, W,
        )
        out_path = out_dir / f"gabor_slope_stripes_sample_{label}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {i}/{len(alignment_powers)}  align_pow={alignment_power}  mean={mean_val:.4f}  ->  {out_path.name}")

    print()
    print("Summary (mean value in [0,1]):")
    print("  align_pow   mean    min    max   filename")
    for alignment_power, mean_val, min_val, max_val, label in results:
        print(f"  {alignment_power:5.1f}       {mean_val:.3f}   {min_val:.3f}   {max_val:.3f}   gabor_slope_stripes_sample_{label}.png")
    print()
    print(f"All images saved in: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
