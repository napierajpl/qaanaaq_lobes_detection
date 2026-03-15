"""Helpers for generating tile index shapefile (e.g. resolving background_train_ids)."""

import logging
from pathlib import Path
from typing import Optional, Set

from src.training.dataloader import (
    load_filtered_tiles,
    create_data_splits,
    get_background_candidates,
    build_extended_train_tiles,
    load_extended_training_tiles,
    get_background_train_ids_from_extended_tiles,
)

logger = logging.getLogger(__name__)


def resolve_background_train_ids(
    extended_path: Optional[Path],
    filtered_path: Optional[Path],
    features_dir: Optional[Path],
    targets_dir: Optional[Path],
    train_split: float = 0.7,
    seed: int = 42,
    white_threshold: float = 0.95,
    n_background_and_augmented: Optional[int] = None,
) -> Optional[Set[str]]:
    """
    Resolve set of tile_ids used as background_train for train_usage column.
    If extended_path exists, read from extended_training_tiles.json.
    Else if filtered_path + features_dir + targets_dir are set, compute from splits and background candidates.
    """
    if extended_path is not None and extended_path.exists():
        tiles_list, _config, _stats = load_extended_training_tiles(extended_path)
        ids = get_background_train_ids_from_extended_tiles(tiles_list)
        logger.info(f"Train usage from extended_training_tiles.json: {len(ids)} background_train")
        return ids

    if filtered_path is None or features_dir is None or targets_dir is None:
        return None
    if not filtered_path.exists():
        logger.warning(f"Filtered tiles not found: {filtered_path}, skipping train_usage")
        return None
    if not features_dir.exists() or not targets_dir.exists():
        logger.warning("Features or targets dir missing, skipping train_usage")
        return None

    all_tiles = load_filtered_tiles(filtered_path)
    valid_ids = {t["tile_id"] for t in all_tiles}
    train_tiles, _, _ = create_data_splits(
        all_tiles,
        train_split=train_split,
        val_split=0.15,
        test_split=0.15,
        random_seed=seed,
    )
    background_candidates = get_background_candidates(
        features_dir, targets_dir, valid_ids, white_threshold=white_threshold
    )
    n_add = min(n_background_and_augmented or len(train_tiles), len(background_candidates))
    extended = build_extended_train_tiles(
        train_tiles, background_candidates, n_add=n_add, random_seed=seed
    )
    background_train_ids = {
        t["tile_id"] for t in extended
        if not t.get("augment") and t["tile_id"] not in valid_ids
    }
    logger.info(f"Train usage: {n_add} background_train tiles (train_usage column + QML)")
    return background_train_ids
