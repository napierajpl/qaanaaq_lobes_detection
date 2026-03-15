#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.path_utils import get_project_root, resolve_path

def main():
    project_root = get_project_root(Path(__file__))
    base = resolve_path(Path("data/processed/tiles/train_512/segmentation"), project_root)
    if not base.exists():
        print("Dir missing:", base)
        return
    tiles = ["tile_5513", "tile_4750", "tile_5005", "tile_6953", "tile_7370"]
    for tid in tiles:
        p = base / f"{tid}.tif"
        if not p.exists():
            print(tid, "missing")
            continue
        with rasterio.open(p) as src:
            d = src.read(1)
            nodata = src.nodata
        d = np.asarray(d, dtype=np.float64)
        valid = d != nodata if nodata is not None else np.ones_like(d, dtype=bool)
        if not np.any(valid):
            print(f"{tid}: all nodata")
            continue
        v = d[valid]
        mn, mx = float(np.min(v)), float(np.max(v))
        uniq = np.unique(v)
        n_uniq = len(uniq)
        # Check if values are just a function of column (gradient-like)
        col_idx = np.arange(d.shape[1], dtype=np.float64)
        row_corr = np.corrcoef(d.mean(axis=0), col_idx)[0, 1] if d.shape[1] > 1 else 0
        col_corr = np.corrcoef(d.mean(axis=1), np.arange(d.shape[0], dtype=np.float64))[0, 1] if d.shape[0] > 1 else 0
        gradient_like = abs(row_corr) > 0.9 or abs(col_corr) > 0.9
        print(f"{tid}: shape={d.shape} nodata={nodata} min={mn} max={mx} n_unique={n_uniq} "
              f"row_corr={row_corr:.3f} col_corr={col_corr:.3f} gradient_like={gradient_like}")

if __name__ == "__main__":
    main()
