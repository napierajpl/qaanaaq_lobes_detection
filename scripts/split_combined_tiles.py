#!/usr/bin/env python3
"""Split multi-band combined feature tiles into per-layer tile directories.

Reads 5-band tiles from features/features_combined/ and writes:
  - rgb/   (bands 1-3)
  - dem/   (band 4)
  - slope/ (band 5)
"""

import argparse
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rasterio

BAND_LAYOUT = [
    ("rgb", [1, 2, 3]),
    ("dem", [4]),
    ("slope", [5]),
]


def _split_tile(src_path: Path, output_dirs: dict[str, Path]) -> str:
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        for layer_name, bands in BAND_LAYOUT:
            out_profile = profile.copy()
            out_profile["count"] = len(bands)
            out_path = output_dirs[layer_name] / src_path.name
            if out_path.exists():
                continue
            data = src.read(bands)
            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(data)
    return src_path.name


def _find_combined_dir(tiles_root: Path) -> Path:
    candidates = [
        tiles_root / "features" / "features_combined",
        *sorted((tiles_root / "features").glob("features_combined*")),
    ]
    for d in candidates:
        if d.is_dir() and any(d.glob("*.tif")):
            return d
    raise FileNotFoundError(
        f"No combined tile directory found under {tiles_root / 'features'}"
    )


def split_combined_tiles(tiles_root: Path, workers: int = 4):
    combined_dir = _find_combined_dir(tiles_root)
    print(f"Found combined tiles in: {combined_dir}")

    tile_files = sorted(combined_dir.glob("*.tif"))
    if not tile_files:
        raise FileNotFoundError(f"No .tif files in {combined_dir}")

    output_dirs = {}
    for layer_name, _ in BAND_LAYOUT:
        d = tiles_root / layer_name
        d.mkdir(parents=True, exist_ok=True)
        output_dirs[layer_name] = d

    print(f"Splitting {len(tile_files)} tiles into {list(output_dirs.keys())}...")

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_split_tile, f, output_dirs): f for f in tile_files
        }
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 500 == 0 or done == len(tile_files):
                print(f"  {done}/{len(tile_files)}")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tiles_root", type=Path,
        help="Root directory of a tile set (e.g. data/processed/tiles/train_512)",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    split_combined_tiles(args.tiles_root, args.workers)


if __name__ == "__main__":
    main()
