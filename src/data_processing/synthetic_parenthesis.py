"""Generate synthetic parenthesis dataset for sanity-check training (512x512)."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio

from src.data_processing.synthetic_shapes import place_random_parentheses_on_tile
from src.training.dataloader import load_filtered_tiles


def generate_synthetic_parenthesis_dataset(
    source_filtered_tiles: Path,
    source_features_dir: Path,
    output_dir: Path,
    tile_size: int,
    max_tiles: Optional[int],
    shapes_per_tile: int,
    shape_height_px: int,
    seed: int,
) -> None:
    output_dir = Path(output_dir)
    features_dir = output_dir / "features"
    targets_dir = output_dir / "targets"
    features_dir.mkdir(parents=True, exist_ok=True)
    targets_dir.mkdir(parents=True, exist_ok=True)

    tiles = load_filtered_tiles(Path(source_filtered_tiles), show_progress=False)
    if max_tiles is not None:
        tiles = tiles[:max_tiles]
    rng = np.random.default_rng(seed)

    out_tiles = []
    for i, tile_info in enumerate(tiles):
        tile_id = tile_info.get("tile_id", f"tile_{i:04d}")
        feat_rel = tile_info.get("features_path", "").replace("\\", "/")
        feat_path = source_features_dir / feat_rel if not Path(feat_rel).is_absolute() else Path(feat_rel)
        if not feat_path.exists():
            continue
        with rasterio.open(feat_path) as src:
            features = src.read()
            profile = src.profile.copy()
        if features.shape[0] < 5:
            continue
        H, W = features.shape[1], features.shape[2]
        if tile_size and (H != tile_size or W != tile_size):
            continue
        target = np.zeros((H, W), dtype=np.float32)
        place_random_parentheses_on_tile(features, target, shape_height_px, shapes_per_tile, rng)

        out_feat = features_dir / f"{tile_id}.tif"
        out_tgt = targets_dir / f"{tile_id}.tif"
        profile.update(count=5, dtype=features.dtype)
        with rasterio.open(out_feat, "w", **profile) as dst:
            dst.write(features)
        profile_t = profile.copy()
        profile_t.update(count=1, dtype=target.dtype)
        with rasterio.open(out_tgt, "w", **profile_t) as dst:
            dst.write(target, 1)
        out_tiles.append({
            "tile_id": tile_id,
            "features_path": f"{tile_id}.tif",
            "targets_path": f"{tile_id}.tif",
        })

    filtered_path = output_dir / "filtered_tiles.json"
    with open(filtered_path, "w", encoding="utf-8") as f:
        json.dump({"tiles": out_tiles, "tile_size": tile_size}, f, indent=2)
    print(f"Wrote {len(out_tiles)} tiles to {output_dir}")
    print(f"  features: {features_dir}")
    print(f"  targets: {targets_dir}")
    print(f"  {filtered_path}")
