"""
One-off: load slope-stripes raster + RGB, print stats, save a side-by-side figure.
Run after create_slope_stripes_channel.py (e.g. on dev crop).
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.utils.path_utils import get_project_root, resolve_path


def main():
    project_root = get_project_root(Path(__file__))
    rgb_path = resolve_path(Path("data/processed/raster/dev/qaanaaq_rgb_0_2m_cropped1024x1024.tif"), project_root)
    stripe_path = resolve_path(Path("data/processed/raster/dev/slope_stripes_channel_cropped1024x1024.tif"), project_root)
    out_path = project_root / "data/processed/raster/dev/slope_stripes_sample.png"

    if not stripe_path.exists():
        print(f"Run create_slope_stripes_channel.py first. Missing: {stripe_path}")
        return 1

    with rasterio.open(stripe_path) as src:
        stripe = src.read(1)
    stripe = np.asarray(stripe, dtype=np.float64)
    stripe = np.clip(stripe, 0.0, 1.0)

    print("Slope-stripes channel stats:")
    print(f"  Shape: {stripe.shape}")
    print(f"  Min:   {float(np.min(stripe)):.4f}")
    print(f"  Max:   {float(np.max(stripe)):.4f}")
    print(f"  Mean:  {float(np.mean(stripe)):.4f}")

    rgb = None
    if rgb_path.exists():
        with rasterio.open(rgb_path) as src:
            rgb = src.read([1, 2, 3])
        rgb = np.transpose(rgb, (1, 2, 0))
        if rgb.max() > 1.5:
            rgb = rgb.astype(np.float64) / 255.0
        rgb = np.clip(rgb, 0, 1)

    fig, axes = plt.subplots(1, 2 if rgb is not None else 1, figsize=(10 if rgb is not None else 5, 5))
    if rgb is not None:
        axes = list(axes)
    else:
        axes = [axes]

    if rgb is not None:
        axes[0].imshow(rgb)
        axes[0].set_title("RGB (dev crop 1024×1024)")
        axes[0].axis("off")

    im = axes[-1].imshow(stripe, cmap="gray", vmin=0, vmax=1)
    axes[-1].set_title("Slope-stripes (coherence × slope alignment)")
    axes[-1].axis("off")
    plt.colorbar(im, ax=axes[-1], shrink=0.7, label="0–1")

    fig.suptitle("Slope-stripes channel sample", fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
