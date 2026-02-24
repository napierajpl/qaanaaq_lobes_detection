"""
Run slope-stripes channel with 10 (sigma_smooth, sigma_structure) combinations,
save sample PNGs for comparison. Each figure: 6 views (RGB, Slope-stripes, DEM, Slope, Aspect, 10x10 mean heatmap) with 10x10 grid.
Output: slope_stripes_sample_smooth{X}_structure{Y}.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio
import matplotlib.pyplot as plt

from src.preprocessing.texture_hints import compute_slope_stripes_channel, aspect_from_dem
from src.utils.path_utils import get_project_root, resolve_path

GRID_SIZE = 10


def _param_to_str(v: float) -> str:
    return str(v).replace(".", "_")


def draw_10x10_grid(ax, H: int, W: int, color: str = "white", alpha: float = 0.7, lw: float = 0.5):
    step_y = H / GRID_SIZE
    step_x = W / GRID_SIZE
    for k in range(1, GRID_SIZE):
        ax.axhline(k * step_y, color=color, alpha=alpha, lw=lw)
        ax.axvline(k * step_x, color=color, alpha=alpha, lw=lw)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)


def compute_cell_means(arr: np.ndarray, n: int = GRID_SIZE) -> np.ndarray:
    H, W = arr.shape
    step_y, step_x = H / n, W / n
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            r0, r1 = int(i * step_y), int((i + 1) * step_y)
            c0, c1 = int(j * step_x), int((j + 1) * step_x)
            out[i, j] = np.mean(arr[r0:r1, c0:c1])
    return out


def main():
    project_root = get_project_root(Path(__file__))
    rgb_path = resolve_path(Path("data/processed/raster/dev/qaanaaq_rgb_0_2m_cropped1024x1024.tif"), project_root)
    dem_path = resolve_path(Path("data/processed/raster/dev/dem_from_arcticDEM_cropped1024x1024_resampled.tif"), project_root)
    slope_path = resolve_path(Path("data/processed/raster/dev/slope_from_dem_cropped1024x1024_resampled.tif"), project_root)
    out_dir = project_root / "data/processed/raster/dev"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not rgb_path.exists():
        print(f"Missing RGB: {rgb_path}")
        return 1
    if not dem_path.exists():
        print(f"Missing DEM: {dem_path}")
        return 1
    if not slope_path.exists():
        print(f"Missing slope: {slope_path}")
        return 1

    with rasterio.open(rgb_path) as src:
        rgb = src.read([1, 2, 3])
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
    with rasterio.open(slope_path) as src:
        slope = src.read(1)
    H, W = dem.shape
    if rgb.shape[1:] != (H, W) or slope.shape != (H, W):
        print("RGB, DEM, slope shape mismatch")
        return 1

    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    dem = np.asarray(dem, dtype=np.float64)
    slope = np.asarray(slope, dtype=np.float64)

    dem_safe = np.nan_to_num(dem, nan=0.0, posinf=0.0, neginf=0.0)
    slope_safe = np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0)
    aspect = aspect_from_dem(dem)
    aspect_display = (aspect.astype(np.float64) + np.pi) / (2.0 * np.pi)
    aspect_display = np.clip(aspect_display, 0.0, 1.0)

    combinations = [
        (0.8, 1.0),
        (1.0, 1.5),
        (1.2, 1.8),
        (1.5, 2.0),
        (1.5, 3.0),
        (2.0, 2.5),
        (2.0, 3.5),
        (2.5, 3.0),
        (3.0, 4.0),
        (3.5, 5.0),
    ]

    print("Running 10 parameter combinations (smooth, structure)...")
    print("Each figure: RGB, Slope-stripes, DEM, Slope, Aspect (from DEM), 10x10 mean heatmap (with 10x10 grid on first 5)")
    print()
    results = []

    for i, (sigma_smooth, sigma_structure) in enumerate(combinations, 1):
        label = f"smooth{_param_to_str(sigma_smooth)}structure{_param_to_str(sigma_structure)}"
        stripe = compute_slope_stripes_channel(
            rgb, dem, sigma_smooth=sigma_smooth, sigma_structure=sigma_structure
        )
        stripe = np.clip(stripe.astype(np.float64), 0.0, 1.0)
        mean_val = float(np.mean(stripe))
        min_val = float(np.min(stripe))
        max_val = float(np.max(stripe))
        results.append((sigma_smooth, sigma_structure, mean_val, min_val, max_val, label))

        cell_means = compute_cell_means(stripe)

        fig = plt.figure(figsize=(14, 14))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.26, wspace=0.12)

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[2, 0])
        ax5 = fig.add_subplot(gs[2, 1])

        rgb_display = np.transpose(np.clip(rgb, 0, 1), (1, 2, 0))
        ax0.imshow(rgb_display)
        ax0.set_title("RGB (1024x1024)")
        ax0.axis("off")
        draw_10x10_grid(ax0, H, W)

        im1 = ax1.imshow(stripe, cmap="gray", vmin=0, vmax=1)
        ax1.set_title(f"Slope-stripes  smooth={sigma_smooth}  structure={sigma_structure}")
        ax1.axis("off")
        draw_10x10_grid(ax1, H, W)
        plt.colorbar(im1, ax=ax1, shrink=0.6, label="0-1")

        im2 = ax2.imshow(dem_safe, cmap="terrain")
        ax2.set_title("DEM")
        ax2.axis("off")
        draw_10x10_grid(ax2, H, W)
        plt.colorbar(im2, ax=ax2, shrink=0.6, label="m")

        im3 = ax3.imshow(slope_safe, cmap="viridis")
        ax3.set_title("Slope (steepness, deg)")
        ax3.axis("off")
        draw_10x10_grid(ax3, H, W)
        plt.colorbar(im3, ax=ax3, shrink=0.6, label="deg")

        im4 = ax4.imshow(aspect_display, cmap="hsv", vmin=0, vmax=1)
        ax4.set_title("Slope orientation")
        ax4.axis("off")
        draw_10x10_grid(ax4, H, W)
        plt.colorbar(im4, ax=ax4, shrink=0.5, label="aspect orientation (cyclic)")

        im5 = ax5.imshow(cell_means, cmap="gray", vmin=0, vmax=1)
        ax5.set_title("Mean slope-stripes per 10x10 cell")
        for ri in range(GRID_SIZE):
            for cj in range(GRID_SIZE):
                v = cell_means[ri, cj]
                ax5.text(cj, ri, f"{v:.2f}", ha="center", va="center", fontsize=6, color="black" if v > 0.5 else "white")
        ax5.set_xticks([])
        ax5.set_yticks([])
        plt.colorbar(im5, ax=ax5, shrink=0.7, label="mean 0-1")

        fig.suptitle(f"slope_stripes_sample_{label}.png", fontsize=12)
        out_path = out_dir / f"slope_stripes_sample_{label}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {i:2d}/10  smooth={sigma_smooth:.1f}  structure={sigma_structure:.1f}  mean={mean_val:.4f}  ->  {out_path.name}")

    print()
    print("Summary (mean value in [0,1]):")
    print("  smooth  structure   mean    min    max   filename")
    for sigma_smooth, sigma_structure, mean_val, min_val, max_val, label in results:
        print(f"  {sigma_smooth:5.1f}   {sigma_structure:5.1f}     {mean_val:.3f}   {min_val:.3f}   {max_val:.3f}   slope_stripes_sample_{label}.png")
    print()
    print(f"All images saved in: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
