"""Resolved training config: paths, layer registry, splits, illumination, normalization."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.training.layer_registry import LayerRegistry, build_layer_registry
from src.utils.config_utils import get_training_path_key
from src.utils.path_utils import resolve_path

logger = logging.getLogger(__name__)


@dataclass
class ResolvedTrainingPaths:
    filtered_tiles_path: Path
    targets_dir: Path
    models_dir: Path
    layer_registry: LayerRegistry
    path_key: str
    tile_size: int
    target_mode: str
    binary_threshold: float


def resolve_training_paths(
    config: dict,
    mode: str,
    project_root: Path,
    filtered_tiles_override: Optional[Path] = None,
) -> ResolvedTrainingPaths:
    if mode == "synthetic_parenthesis":
        config.setdefault("data", {})["use_background_and_augmentation"] = False

    tile_size = config["data"].get("tile_size", 256)
    target_mode = (config.get("target_mode") or "proximity").lower()
    binary_threshold = float(config.get("binary_threshold", 1.0))
    path_key = get_training_path_key(mode, tile_size)
    paths = config["paths"][path_key]

    filtered_tiles_path = resolve_path(Path(paths["filtered_tiles"]), project_root)
    if filtered_tiles_override is not None:
        filtered_tiles_path = resolve_path(Path(filtered_tiles_override), project_root)

    targets_dir = resolve_path(Path(paths["targets_dir"]), project_root)
    models_dir = resolve_path(Path(paths["models_dir"]), project_root)

    layer_registry = build_layer_registry(config, project_root, path_key)

    return ResolvedTrainingPaths(
        filtered_tiles_path=filtered_tiles_path,
        targets_dir=targets_dir,
        models_dir=models_dir,
        layer_registry=layer_registry,
        path_key=path_key,
        tile_size=tile_size,
        target_mode=target_mode,
        binary_threshold=binary_threshold,
    )


def validate_data_splits(
    train_split: float, val_split: float, test_split: float
) -> None:
    split_sum = train_split + val_split + test_split
    if abs(split_sum - 1.0) >= 1e-6:
        raise ValueError(
            f"Data splits must sum to 1.0, got {split_sum:.6f}. "
            f"train={train_split}, val={val_split}, test={test_split}"
        )


def apply_illumination_filter(
    train_tiles: List[dict],
    val_tiles: List[dict],
    test_tiles: List[dict],
    illumination_filter: str,
    illumination_include_background: bool,
) -> tuple:
    if illumination_filter not in ("shadow", "sun"):
        return train_tiles, val_tiles, test_tiles

    def matches(t: dict) -> bool:
        if illumination_include_background and t.get("role") == "background":
            return True
        return t.get("illumination") == illumination_filter

    train_filtered = [t for t in train_tiles if matches(t)]
    val_filtered = [t for t in val_tiles if matches(t)]
    test_filtered = [t for t in test_tiles if matches(t)]
    return train_filtered, val_filtered, test_filtered


def validate_in_channels(config: dict) -> int:
    """Compute in_channels from the layers config and warn if YAML model.in_channels differs."""
    layers_config = config.get("layers", {})
    computed = sum(
        lc["bands"]
        for lc in layers_config.values()
        if lc.get("enabled", True)
    )
    if computed < 1:
        raise ValueError(
            "At least one input layer must be enabled. "
            "Set at least one layer's enabled: true in the layers config."
        )
    yaml_value = config.get("model", {}).get("in_channels")
    if yaml_value is not None and int(yaml_value) != computed:
        logger.warning(
            "model.in_channels in YAML is %s but enabled layers compute %s channels. "
            "Using computed value %s.",
            yaml_value, computed, computed,
        )
    return computed


def log_training_config_summary(resolved: ResolvedTrainingPaths, mode: str) -> None:
    r = resolved
    reg = r.layer_registry
    logger.info("=== Training Configuration ===")
    logger.info(
        "Mode: %s (tile size: %sx%s), target_mode: %s",
        mode, r.tile_size, r.tile_size, r.target_mode,
    )
    logger.info("Filtered tiles: %s", r.filtered_tiles_path)
    logger.info("Targets dir: %s", r.targets_dir)
    logger.info("Models dir: %s", r.models_dir)
    enabled_names = reg.layer_names()
    logger.info(
        "Enabled layers: %s -> %s channels",
        ", ".join(enabled_names), reg.in_channels,
    )
    for layer in reg.enabled_layers:
        logger.info("  %s: %s (%s bands, norm=%s)",
                     layer.spec.name, layer.tile_dir,
                     layer.spec.bands, layer.spec.normalization)


def get_normalization_stats(
    train_tiles: List[dict],
    layer_registry: LayerRegistry,
    use_bg_aug: bool,
    extended_path: Path,
) -> Dict[str, Dict[str, float]]:
    needs_stats = any(
        l.spec.normalization == "standardize"
        for l in layer_registry.enabled_layers
    )
    if not needs_stats:
        return {}

    logger.info("Computing normalization statistics...")
    extended_loaded = use_bg_aug and extended_path.exists()
    tiles_for_stats = (
        [t for t in train_tiles if t.get("role") == "lobe"]
        if extended_loaded
        else train_tiles
    )
    tile_ids = [t.get("tile_id", "") for t in tiles_for_stats if t.get("tile_id")]
    stats = layer_registry.compute_normalization_stats(tile_ids)
    for name, s in stats.items():
        logger.info(
            "%s stats: mean=%.2f, std=%.2f",
            name, s.get("mean", 0), s.get("std", 0),
        )
    return stats
