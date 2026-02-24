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
import matplotlib.pyplot as plt

from src.preprocessing.texture_hints import (
    compute_slope_stripes_channel,
    compute_gabor_slope_stripes_channel,
    aspect_from_dem,
)
from src.utils.path_utils import get_project_root, resolve_path

GRID_SIZE = 10


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


def make_six_panel(
    rgb: np.ndarray,
    stripe: np.ndarray,
    dem_safe: np.ndarray,
    slope_safe: np.ndarray,
    aspect_display: np.ndarray,
    method_title: str,
    stripe_panel_title: str,
    H: int,
    W: int,
) -> plt.Figure:
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
    ax0.set_title("RGB rotated 90 deg (1024x1024)")
    ax0.axis("off")
    draw_10x10_grid(ax0, H, W)

    im1 = ax1.imshow(stripe, cmap="gray", vmin=0, vmax=1)
    ax1.set_title(stripe_panel_title)
    ax1.axis("off")
    draw_10x10_grid(ax1, H, W)
    plt.colorbar(im1, ax=ax1, shrink=0.6, label="0-1")

    im2 = ax2.imshow(dem_safe, cmap="terrain")
    ax2.set_title("DEM (unchanged)")
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

    fig.suptitle(f"{method_title}  |  RGB rotated 90°  (expect low values)", fontsize=12)
    return fig


def main():
    project_root = get_project_root(Path(__file__))
    rgb_path = resolve_path(Path("data/processed/raster/dev/qaanaaq_rgb_0_2m_cropped1024x1024.tif"), project_root)
    dem_path = resolve_path(Path("data/processed/raster/dev/dem_from_arcticDEM_cropped1024x1024_resampled.tif"), project_root)
    slope_path = resolve_path(Path("data/processed/raster/dev/slope_from_dem_cropped1024x1024_resampled.tif"), project_root)
    out_dir = project_root / "data/processed/raster/dev"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not rgb_path.exists() or not dem_path.exists() or not slope_path.exists():
        print("Missing dev rasters. Run prepare_training_data --dev first.")
        return 1

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

    print("Sanity check: RGB rotated 90°, DEM unchanged. Expect low slope-stripes values.")
    print()

    stripe_st = compute_slope_stripes_channel(rgb_rot, dem, sigma_smooth=1.5, sigma_structure=3.0)
    stripe_st = np.clip(stripe_st.astype(np.float64), 0.0, 1.0)
    mean_st = float(np.mean(stripe_st))
    print(f"Structure tensor (smooth=1.5, structure=3.0):  mean = {mean_st:.4f}")

    fig_st = make_six_panel(
        rgb_rot,
        stripe_st,
        dem_safe,
        slope_safe,
        aspect_display,
        "Structure tensor",
        "Structure tensor slope-stripes  (smooth=1.5  structure=3.0)",
        H,
        W,
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
        rgb_rot,
        stripe_gab,
        dem_safe,
        slope_safe,
        aspect_display,
        "Gabor",
        "Gabor slope-stripes  (freq=0.15  sigma=5.0)",
        H,
        W,
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
