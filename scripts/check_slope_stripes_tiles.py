#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.path_utils import get_project_root, resolve_path

def main():
    project_root = get_project_root(Path(__file__))
    base = resolve_path(Path("data/processed/tiles/train_512/slope_stripes_channel"), project_root)
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
        mn, mx, me = float(np.nanmin(d)), float(np.nanmax(d)), float(np.nanmean(d))
        ok = "OK" if mx > 0 else "ALL ZEROS"
        print(f"{tid}: min={mn:.4f} max={mx:.4f} mean={me:.4f}  {ok}")

if __name__ == "__main__":
    main()
