"""
Generate synthetic parenthesis dataset for sanity-check training (512x512 only).

Uses existing RGB (+ DEM + Slope) tiles, draws big black "(" and ")" at random
positions/rotations, burns them into RGB, and writes target raster (20 inside shape, 0 outside).
Output layout matches normal tiles: features/, targets/, filtered_tiles.json.
Synthetic mode is 512x512 only. Train with: --mode synthetic_parenthesis
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio

from src.training.dataloader import load_filtered_tiles
from src.utils.path_utils import get_project_root, resolve_path


def _make_parenthesis_mask(
    char: str,
    height_px: int,
    font_path: str | None = None,
) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise ImportError("PIL is required for generate_synthetic_parenthesis_dataset. Install: pip install Pillow")

    font_size = height_px
    if font_path and Path(font_path).exists():
        font = ImageFont.truetype(font_path, font_size)
    else:
        for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            try:
                font = ImageFont.truetype(name, font_size)
                break
            except OSError:
                continue
        else:
            font = ImageFont.load_default()

    img = Image.new("L", (font_size * 2, font_size * 2), 0)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), char, fill=255, font=font)
    arr = np.array(img)
    mask = (arr > 0).astype(np.uint8)
    r, c = np.where(mask)
    if r.size == 0:
        return np.zeros((height_px, height_px), dtype=np.uint8)
    rmin, rmax, cmin, cmax = r.min(), r.max(), c.min(), c.max()
    mask = mask[rmin : rmax + 1, cmin : cmax + 1]
    return mask


def _rotate_mask(mask: np.ndarray, angle_deg: float) -> np.ndarray:
    from scipy.ndimage import rotate
    rotated = rotate(mask.astype(float), angle_deg, order=0, reshape=True, mode="constant", cval=0)
    return (rotated > 0.5).astype(np.uint8)


def _place_shapes_on_tile(
    features: np.ndarray,
    target: np.ndarray,
    shape_height_px: int,
    shapes_per_tile: int,
    rng: np.random.Generator,
) -> None:
    H, W = target.shape
    max_side = min(H, W) - 10
    use_height = min(shape_height_px, max_side) if max_side > 0 else shape_height_px
    chars = ["(", ")"]
    masks = [_make_parenthesis_mask(c, use_height) for c in chars]
    for _ in range(shapes_per_tile):
        idx = rng.integers(0, len(masks))
        mask = masks[idx].copy()
        angle = rng.uniform(0, 360)
        mask = _rotate_mask(mask, angle)
        mh, mw = mask.shape
        if mh > H or mw > W:
            from scipy.ndimage import zoom
            sy, sx = (H - 2) / mh, (W - 2) / mw
            scale = min(sy, sx, 1.0)
            if scale < 1.0:
                mask = zoom(mask.astype(float), scale, order=0)
                mask = (mask > 0.5).astype(np.uint8)
            mh, mw = mask.shape
        if mh > H or mw > W:
            continue
        top = rng.integers(0, max(1, H - mh + 1))
        left = rng.integers(0, max(1, W - mw + 1))
        roi_r = slice(top, top + mh)
        roi_c = slice(left, left + mw)
        where = mask > 0
        features[0:3, roi_r, roi_c][:, where] = 0
        # Target: 20 inside parenthesis, 0 outside (same 0–20 range as real proximity; binary, not distance transform)
        target[roi_r, roi_c] = np.where(where, 20, target[roi_r, roi_c])


def generate_dataset(
    source_filtered_tiles: Path,
    source_features_dir: Path,
    output_dir: Path,
    tile_size: int,
    max_tiles: int | None,
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
        tiles = tiles[: max_tiles]
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
        # Binary target: 0 outside, 20 inside shapes (we do not use generate_proximity_map.py here)
        target = np.zeros((H, W), dtype=np.float32)
        _place_shapes_on_tile(features, target, shape_height_px, shapes_per_tile, rng)

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


def main() -> None:
    project_root = get_project_root(Path(__file__))
    parser = argparse.ArgumentParser(
        description="Generate synthetic parenthesis dataset from existing tiles.",
    )
    parser.add_argument(
        "--source-filtered-tiles",
        type=Path,
        default=project_root / "data/processed/tiles/dev/train_512/filtered_tiles.json",
        help="Path to source filtered_tiles.json (512 tiles)",
    )
    parser.add_argument(
        "--source-features-dir",
        type=Path,
        default=project_root / "data/processed/tiles/dev/train_512/features",
        help="Base dir for source 512x512 feature rasters",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data/processed/tiles/synthetic_parenthesis_512",
        help="Output directory (features/, targets/, filtered_tiles.json); use synthetic_parenthesis_512 for training",
    )
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size (512 only for synthetic mode)")
    parser.add_argument("--max-tiles", type=int, default=None, help="Max tiles to generate (default: all)")
    parser.add_argument("--shapes-per-tile", type=int, default=2, help="Number of ( ) shapes per tile")
    parser.add_argument("--shape-height-px", type=int, default=300, help="Height of parenthesis in px")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_filtered = resolve_path(args.source_filtered_tiles, project_root)
    source_features = resolve_path(args.source_features_dir, project_root)
    output_dir = resolve_path(args.output_dir, project_root)

    if not source_filtered.exists():
        raise FileNotFoundError(f"Source filtered tiles not found: {source_filtered}")
    if not source_features.exists():
        raise FileNotFoundError(f"Source features dir not found: {source_features}")

    generate_dataset(
        source_filtered_tiles=source_filtered,
        source_features_dir=source_features,
        output_dir=output_dir,
        tile_size=args.tile_size,
        max_tiles=args.max_tiles,
        shapes_per_tile=args.shapes_per_tile,
        shape_height_px=args.shape_height_px,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
