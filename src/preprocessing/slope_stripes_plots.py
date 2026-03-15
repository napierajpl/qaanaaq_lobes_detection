"""Shared plotting helpers for slope-stripes sanity and parameter-sweep scripts."""

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

GRID_SIZE = 10


def draw_10x10_grid(ax, H: int, W: int, color: str = "white", alpha: float = 0.7, lw: float = 0.5) -> None:
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
    stripe_panel_title: str,
    suptitle: str,
    H: int,
    W: int,
    rgb_title: str = "RGB (1024x1024)",
    dem_title: str = "DEM",
) -> "plt.Figure":
    if plt is None:
        raise ImportError("matplotlib is required for slope_stripes_plots")
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
    ax0.set_title(rgb_title)
    ax0.axis("off")
    draw_10x10_grid(ax0, H, W)

    im1 = ax1.imshow(stripe, cmap="gray", vmin=0, vmax=1)
    ax1.set_title(stripe_panel_title)
    ax1.axis("off")
    draw_10x10_grid(ax1, H, W)
    plt.colorbar(im1, ax=ax1, shrink=0.6, label="0-1")

    im2 = ax2.imshow(dem_safe, cmap="terrain")
    ax2.set_title(dem_title)
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
            ax5.text(
                cj, ri, f"{v:.2f}",
                ha="center", va="center", fontsize=6,
                color="black" if v > 0.5 else "white",
            )
    ax5.set_xticks([])
    ax5.set_yticks([])
    plt.colorbar(im5, ax=ax5, shrink=0.7, label="mean 0-1")

    fig.suptitle(suptitle, fontsize=12)
    return fig


def param_to_str(v: float) -> str:
    return str(v).replace(".", "_")
