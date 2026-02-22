#!/usr/bin/env python3
"""
Prepare extended training set: add background tiles and write augmented lobe tiles to disk.

Run this after filtering (filter_tiles.py) and before training. Produces:
  - features/<augmented_subdir>/<tile_id>_aug_<k>.tif (4 rotations + contrast/saturation per lobe)
  - targets/<augmented_subdir>/<tile_id>_aug_<k>.tif
  - extended_training_tiles.json (train list with role: lobe | background | augmented_lobe)

Training then loads extended_training_tiles.json and uses it as the train set (no on-the-fly augmentation).
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.transform import from_bounds
import torch
from torchvision.transforms.functional import adjust_contrast, adjust_saturation
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.dataloader import (
    load_filtered_tiles,
    create_data_splits,
    get_background_candidates,
    save_extended_training_tiles,
)
from src.utils.path_utils import get_project_root, resolve_path


AUGMENTED_SUBDIR = "augmented"


def _apply_contrast_saturation_rgb(rgb: np.ndarray, c: float, s: float) -> np.ndarray:
    """Apply contrast and saturation to RGB (C, H, W), values 0-255. Returns same shape, 0-255."""
    t = torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)
    t = adjust_contrast(t, c)
    t = adjust_saturation(t, s)
    out = (t.squeeze(0).numpy() * 255.0).clip(0, 255)
    return out.astype(rgb.dtype)


def _write_augmented_tile(
    features_arr: np.ndarray,
    target_arr: np.ndarray,
    feat_path: Path,
    tgt_path: Path,
    profile_feat: dict,
    profile_tgt: dict,
    k_rot: int,
    c: float,
    s: float,
) -> None:
    """Rotate features and target by k_rot (0,1,2,3), apply c/s to RGB; write to feat_path, tgt_path."""
    features = np.array(features_arr)
    target = np.array(target_arr)
    if k_rot > 0:
        features = np.rot90(features, k_rot, axes=(1, 2))
        target = np.rot90(target, k_rot, axes=(0, 1))
    features[0:3] = _apply_contrast_saturation_rgb(features[0:3], c, s)
    h, w = features.shape[1], features.shape[2]
    transform = from_bounds(0, 0, w, h, w, h)
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    tgt_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        feat_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=features.shape[0],
        dtype=features.dtype,
        crs=profile_feat.get("crs"),
        transform=transform,
        nodata=profile_feat.get("nodata"),
        compress="lzw",
    ) as dst:
        dst.write(features)
    with rasterio.open(
        tgt_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=target.dtype,
        crs=profile_tgt.get("crs"),
        transform=transform,
        nodata=profile_tgt.get("nodata"),
        compress="lzw",
    ) as dst:
        dst.write(target, 1)


def _load_config(config_path: Path, project_root: Path) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for key in ("paths", "splits", "background", "augmentation"):
        cfg.setdefault(key, {})
    return cfg


def main():
    import argparse

    project_root = get_project_root(Path(__file__))

    parser = argparse.ArgumentParser(
        description="Prepare extended training set: sample background tiles and write augmented lobe tiles to disk."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to data_preparation_config.yaml (default: configs/data_preparation_config.yaml)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        choices=[256, 512],
        default=None,
        help="Override config: 256 or 512 (selects paths_256 / paths_512). Default from config is 256.",
    )
    parser.add_argument("--filtered-tiles", type=Path, default=None, help="Override config: filtered_tiles.json")
    parser.add_argument("--features-dir", type=Path, default=None, help="Override config: features directory")
    parser.add_argument("--targets-dir", type=Path, default=None, help="Override config: targets directory")
    parser.add_argument("--output-json", type=Path, default=None, help="Override config: output JSON path")
    parser.add_argument("--white-threshold", type=float, default=None)
    parser.add_argument("--n-background-and-augmented", type=int, default=None)
    parser.add_argument("--train-split", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--contrast-range", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    parser.add_argument("--saturation-range", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    parser.add_argument("--background-ratio", type=float, default=None, help="Override config: background ratio to lobe+aug (e.g. 1.0 = 1:1)")
    args = parser.parse_args()

    config_path = resolve_path(args.config or Path("configs/data_preparation_config.yaml"), project_root)
    if config_path.exists():
        cfg = _load_config(config_path, project_root)
        print(f"Config: {config_path}")
        tile_size = args.tile_size if args.tile_size is not None else cfg.get("tile_size", 256)
        paths_256 = cfg.get("paths_256") or {}
        paths_512 = cfg.get("paths_512") or {}
        legacy_paths = cfg.get("paths") or {}
        if tile_size == 512 and paths_512:
            paths = paths_512
        elif tile_size == 256 and paths_256:
            paths = paths_256
        else:
            paths = legacy_paths or paths_256 or paths_512
        splits = cfg["splits"]
        background = cfg["background"]
        augmentation = cfg["augmentation"]
    else:
        cfg = {"paths": {}, "splits": {}, "background": {}, "augmentation": {}}
        paths = cfg["paths"]
        splits = background = augmentation = {}
        tile_size = args.tile_size or 256

    default_filtered = "data/processed/tiles/train_512/filtered_tiles.json" if tile_size == 512 else "data/processed/tiles/train/filtered_tiles.json"
    default_features = "data/processed/tiles/train_512/features" if tile_size == 512 else "data/processed/tiles/train/features"
    default_targets = "data/processed/tiles/train_512/targets" if tile_size == 512 else "data/processed/tiles/train/targets"
    print(f"Tile size: {tile_size}x{tile_size}")
    filtered_path = resolve_path(
        args.filtered_tiles or Path(paths.get("filtered_tiles", default_filtered)),
        project_root,
    )
    features_dir = resolve_path(
        args.features_dir or Path(paths.get("features_dir", default_features)),
        project_root,
    )
    targets_dir = resolve_path(
        args.targets_dir or Path(paths.get("targets_dir", default_targets)),
        project_root,
    )
    output_json_raw = args.output_json or paths.get("output_json")
    output_json = resolve_path(Path(output_json_raw), project_root) if output_json_raw else filtered_path.parent / "extended_training_tiles.json"

    white_threshold = args.white_threshold if args.white_threshold is not None else background.get("white_threshold", 0.95)
    n_augment_cap = args.n_background_and_augmented if args.n_background_and_augmented is not None else augmentation.get("n_lobe_tiles_to_augment")
    train_split = args.train_split if args.train_split is not None else splits.get("train_split", 0.7)
    seed = args.seed if args.seed is not None else splits.get("seed", 42)
    contrast_range = args.contrast_range or augmentation.get("contrast_range", [0.8, 1.2])
    saturation_range = args.saturation_range or augmentation.get("saturation_range", [0.8, 1.2])
    background_ratio = args.background_ratio if args.background_ratio is not None else background.get("ratio", 1.0)

    if not filtered_path.exists():
        print(f"Error: {filtered_path} not found")
        sys.exit(1)
    if not features_dir.exists() or not targets_dir.exists():
        print("Error: features or targets dir not found")
        sys.exit(1)

    rng = random.Random(seed)
    print("Loading filtered tiles...")
    all_tiles = load_filtered_tiles(filtered_path, show_progress=True)
    print(f"  Loaded {len(all_tiles)} valid tiles.")
    valid_ids = {t["tile_id"] for t in all_tiles}
    val_split = splits.get("val_split", 0.15)
    test_split = splits.get("test_split", 0.15)
    train_tiles, val_tiles, test_tiles = create_data_splits(
        all_tiles, train_split=train_split, val_split=val_split, test_split=test_split, random_seed=seed
    )
    print(f"  Train: {len(train_tiles)}, val: {len(val_tiles)}, test: {len(test_tiles)}.")
    print("Listing tile IDs (features + targets)...")
    background_candidates = get_background_candidates(
        features_dir, targets_dir, valid_ids, white_threshold=white_threshold, show_progress=True
    )
    print(f"  Found {len(background_candidates)} background candidates.")
    n_lobe = len(train_tiles)
    n_augment_sources = min(n_augment_cap or n_lobe, n_lobe)
    n_augmented_entries = 4 * n_augment_sources
    lobe_plus_aug = n_lobe + n_augmented_entries
    n_background_target = int(round(background_ratio * lobe_plus_aug))
    n_background = min(n_background_target, len(background_candidates))
    if n_background == 0 and n_augment_sources == 0:
        print("No background candidates and no augmentation; extended set = train only (no new files)")
        extended_train = [dict(t) for t in train_tiles]
        for t in extended_train:
            t.setdefault("role", "lobe")
        save_extended_training_tiles(
            output_json,
            extended_train,
            config={"n_add": 0},
            stats={"extended_train_size": len(extended_train)},
        )
        print(f"Wrote {output_json}")
        return

    background_sample = [dict(t) for t in rng.sample(background_candidates, n_background)]
    for t in background_sample:
        t["role"] = "background"
    lobe_for_aug = rng.sample(train_tiles, n_augment_sources)

    def resolve_path_str(base: Path, rel: str) -> Path:
        p = base / rel.replace("\\", "/")
        return p if p.exists() else list(base.glob(f"**/{Path(rel).name}"))[0]

    n_augmented_files = n_augmented_entries
    augmented_entries = []
    pbar = tqdm(total=n_augmented_files, desc="Writing augmented tiles", unit="tile")
    for tile in lobe_for_aug:
        tile_id = tile["tile_id"]
        feat_path = resolve_path_str(features_dir, tile["features_path"])
        tgt_path = resolve_path_str(targets_dir, tile["targets_path"])
        with rasterio.open(feat_path) as src:
            features = src.read()
            profile_feat = {"crs": src.crs, "transform": src.transform, "nodata": src.nodata}
        with rasterio.open(tgt_path) as src:
            target = src.read(1)
            profile_tgt = {"crs": src.crs, "transform": src.transform, "nodata": src.nodata}
        c = rng.uniform(*contrast_range)
        s = rng.uniform(*saturation_range)
        for k in range(4):
            aug_id = f"{tile_id}_aug_{k}"
            feat_out = features_dir / AUGMENTED_SUBDIR / f"{aug_id}.tif"
            tgt_out = targets_dir / AUGMENTED_SUBDIR / f"{aug_id}.tif"
            _write_augmented_tile(
                features, target, feat_out, tgt_out, profile_feat, profile_tgt, k, c, s
            )
            pbar.update(1)
            rel_feat = str(Path(AUGMENTED_SUBDIR) / f"{aug_id}.tif")
            rel_tgt = str(Path(AUGMENTED_SUBDIR) / f"{aug_id}.tif")
            augmented_entries.append({
                "tile_id": aug_id,
                "features_path": rel_feat,
                "targets_path": rel_tgt,
                "role": "augmented_lobe",
            })
    pbar.close()

    lobe_with_role = [dict(t) for t in train_tiles]
    for t in lobe_with_role:
        t["role"] = "lobe"
    extended_train = lobe_with_role + background_sample + augmented_entries

    save_extended_training_tiles(
        output_json,
        extended_train,
        config={
            "white_threshold": white_threshold,
            "background_ratio": background_ratio,
            "n_augment_sources": n_augment_sources,
            "random_seed": seed,
            "augmented_subdir": AUGMENTED_SUBDIR,
            "contrast_range": contrast_range,
            "saturation_range": saturation_range,
        },
        stats={
            "total_lobe_train": n_lobe,
            "n_background": n_background,
            "n_augmented": len(augmented_entries),
            "extended_train_size": len(extended_train),
        },
    )
    n_extended = len(extended_train)
    print("\n=== Dataset size (training with augmentation) ===")
    print(f"  Original train (lobe only):       {n_lobe}")
    print(f"  + Augmented lobe (4×{n_augment_sources}):     {len(augmented_entries)}")
    print(f"  + Background (1:1 with lobe+aug): {n_background}")
    if n_background < n_background_target:
        print(f"    (capped: requested {n_background_target}, only {len(background_candidates)} candidates)")
    print(f"  ----------------------------------------")
    print(f"  Extended train size:              {n_extended}  ({n_extended / n_lobe:.2f}× original)")
    print(f"  → Training will see {n_extended} samples per epoch (was {n_lobe} without extended set).")
    print(f"\nWrote {output_json}")


if __name__ == "__main__":
    main()
