#!/usr/bin/env python3
"""Generate shapefile from tile registry for QGIS visualization."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.map_overlays.tile_registry import TileRegistry
from src.map_overlays.shapefile_generator import generate_tile_index_shapefile
from src.training.dataloader import (
    load_filtered_tiles,
    create_data_splits,
    get_background_candidates,
    build_extended_train_tiles,
    load_extended_training_tiles,
    get_background_train_ids_from_extended_tiles,
)
from src.utils.path_utils import get_project_root, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    """Generate shapefile from tile registry."""
    import argparse

    project_root = get_project_root(Path(__file__))

    parser = argparse.ArgumentParser(
        description="Generate shapefile from tile registry for QGIS. "
        "Use --tile-size 512 for 512x512 tiles (registry must exist in train_512/). "
        "Use --filtered-tiles + --features-dir + --targets-dir to add train_usage (incl. background_train)."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Path to tile_registry.json (default: from --tile-size; train/ or train_512/)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        choices=[256, 512],
        default=256,
        help="Tile size: 256 or 512. Sets default registry/output to train/ or train_512/ (default: 256)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output shapefile path (default: same dir as registry, named tile_index.shp)",
    )
    parser.add_argument(
        "--valid-only",
        action="store_true",
        help="Only include valid tiles (filtered tiles)",
    )
    parser.add_argument(
        "--extended-tiles",
        type=Path,
        default=None,
        help="Path to extended_training_tiles.json (enables train_usage from that run; overrides --filtered-tiles)",
    )
    parser.add_argument(
        "--filtered-tiles",
        type=Path,
        default=None,
        help="Path to filtered_tiles.json (enables train_usage by recomputing; use with --features-dir and --targets-dir)",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=None,
        help="Features tile dir (for background_train_ids when --filtered-tiles is set)",
    )
    parser.add_argument(
        "--targets-dir",
        type=Path,
        default=None,
        help="Targets tile dir (for background_train_ids when --filtered-tiles is set)",
    )
    parser.add_argument(
        "--white-threshold",
        type=float,
        default=0.95,
        help="Exclude background candidates with >= this fraction white pixels (default: 0.95)",
    )
    parser.add_argument(
        "--n-background-and-augmented",
        type=int,
        default=None,
        help="Cap for background tiles to add (default: same as train size). Used only for train_usage.",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Train split fraction for computing n_add (default: 0.7)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling background tiles (default: 42)",
    )

    args = parser.parse_args()

    if args.registry is not None:
        registry_path = resolve_path(args.registry, project_root)
    else:
        subdir = "train_512" if args.tile_size == 512 else "train"
        registry_path = project_root / "data/processed/tiles" / subdir / "tile_registry.json"

    if args.output:
        output_path = resolve_path(args.output, project_root)
    else:
        output_path = registry_path.parent / "tile_index.shp"

    # Validate inputs
    if not registry_path.exists():
        logger.error(f"Registry file not found: {registry_path}")
        sys.exit(1)

    # Resolve extended-tiles or filtered/features/targets for train_usage
    extended_path = resolve_path(args.extended_tiles, project_root) if args.extended_tiles else None
    filtered_path = args.filtered_tiles
    features_dir = args.features_dir
    targets_dir = args.targets_dir
    if filtered_path is not None and (features_dir is None or targets_dir is None):
        parent = registry_path.parent
        features_dir = features_dir or parent / "features"
        targets_dir = targets_dir or parent / "targets"
        filtered_path = resolve_path(filtered_path, project_root)
    elif filtered_path is not None:
        filtered_path = resolve_path(filtered_path, project_root)
        features_dir = resolve_path(features_dir, project_root) if features_dir else None
        targets_dir = resolve_path(targets_dir, project_root) if targets_dir else None

    background_train_ids = None
    if extended_path is not None and extended_path.exists():
        tiles_list, _config, _stats = load_extended_training_tiles(extended_path)
        background_train_ids = get_background_train_ids_from_extended_tiles(tiles_list)
        logger.info(f"Train usage from extended_training_tiles.json: {len(background_train_ids)} background_train")
    elif filtered_path is not None and features_dir is not None and targets_dir is not None:
        if not filtered_path.exists():
            logger.warning(f"Filtered tiles not found: {filtered_path}, skipping train_usage")
        elif not features_dir.exists() or not targets_dir.exists():
            logger.warning("Features or targets dir missing, skipping train_usage")
        else:
            all_tiles = load_filtered_tiles(filtered_path)
            valid_ids = {t["tile_id"] for t in all_tiles}
            train_tiles, _, _ = create_data_splits(
                all_tiles,
                train_split=args.train_split,
                val_split=0.15,
                test_split=0.15,
                random_seed=args.seed,
            )
            background_candidates = get_background_candidates(
                features_dir, targets_dir, valid_ids, white_threshold=args.white_threshold
            )
            n_add = min(args.n_background_and_augmented or len(train_tiles), len(background_candidates))
            extended = build_extended_train_tiles(
                train_tiles, background_candidates, n_add=n_add, random_seed=args.seed
            )
            background_train_ids = {
                t["tile_id"] for t in extended
                if not t.get("augment") and t["tile_id"] not in valid_ids
            }
            logger.info(f"Train usage: {n_add} background_train tiles (train_usage column + QML)")

    # Load registry
    logger.info(f"Loading tile registry: {registry_path}")
    registry = TileRegistry(registry_path)

    # Generate shapefile
    generate_tile_index_shapefile(
        registry=registry,
        output_path=output_path,
        include_all_tiles=not args.valid_only,
        background_train_ids=background_train_ids,
    )

    logger.info(f"Shapefile generated successfully: {output_path}")


if __name__ == "__main__":
    main()
