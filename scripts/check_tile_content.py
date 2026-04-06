#!/usr/bin/env python3
"""Report how many tiles have target (lobe) content in filtered_tiles.json.
Uses the same split logic as training so you can see train/val/test content counts."""

import sys
from pathlib import Path

import numpy as np
import rasterio

from src.training.dataloader import create_data_splits, load_filtered_tiles
from src.utils.path_utils import get_project_root, resolve_path


def count_tiles_with_content(
    tile_list: list,
    targets_dir: Path,
    lobe_threshold: float = 0.5,
) -> tuple[int, int]:
    """Return (n_with_content, n_total)."""
    targets_dir = Path(targets_dir)
    n_total = 0
    n_with = 0
    for t in tile_list:
        tid = t.get("tile_id", "")
        if not tid:
            tid = Path(t.get("features_path", "")).stem
        p = targets_dir / f"{tid}.tif"
        if not p.exists():
            continue
        n_total += 1
        with rasterio.open(p) as src:
            data = src.read(1)
        if (data > lobe_threshold).any():
            n_with += 1
    return n_with, n_total


def main() -> None:
    project_root = get_project_root(Path(__file__))
    parser = __import__("argparse").ArgumentParser(
        description="Count tiles with target content (same split as training).",
    )
    parser.add_argument(
        "--filtered",
        type=Path,
        required=True,
        help="Path to filtered_tiles.json",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        required=True,
        help="Path to targets dir (e.g. .../synthetic_parenthesis_256/targets)",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Cap total tiles before split (same as train_model --max-tiles)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--lobe-threshold",
        type=float,
        default=0.5,
        help="Target value above which pixel counts as lobe (default 0.5 for binary)",
    )
    args = parser.parse_args()

    filtered_path = resolve_path(args.filtered, project_root)
    targets_dir = resolve_path(args.targets, project_root)
    if not filtered_path.exists():
        print(f"Error: {filtered_path} not found")
        sys.exit(1)
    if not targets_dir.is_dir():
        print(f"Error: {targets_dir} not found or not a directory")
        sys.exit(1)

    all_tiles = load_filtered_tiles(filtered_path)
    if args.max_tiles is not None:
        rng = np.random.default_rng(args.seed)
        n_take = min(args.max_tiles, len(all_tiles))
        indices = rng.choice(len(all_tiles), size=n_take, replace=False)
        all_tiles = [all_tiles[i] for i in indices]
        print(f"Using random {len(all_tiles)} tiles (--max-tiles {args.max_tiles}, seed {args.seed})")
    train_tiles, val_tiles, test_tiles = create_data_splits(
        all_tiles,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        random_seed=args.seed,
    )

    for label, tiles in [("Train", train_tiles), ("Val", val_tiles), ("Test", test_tiles)]:
        n_with, n_total = count_tiles_with_content(
            tiles, targets_dir, lobe_threshold=args.lobe_threshold
        )
        pct = 100 * n_with / n_total if n_total else 0
        print(f"{label}: {n_with}/{n_total} tiles with target content ({pct:.1f}%)")
    if len(val_tiles) > 0:
        n_val_with, _ = count_tiles_with_content(val_tiles, targets_dir, args.lobe_threshold)
        if n_val_with == 0:
            print("\nWarning: no validation tiles have lobe content. Val metrics (IoU, target stats) will be trivial.")


if __name__ == "__main__":
    main()
