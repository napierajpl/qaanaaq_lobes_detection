"""
Resolved training config: paths, channel count, splits, illumination, normalization.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.utils.config_utils import get_training_path_key
from src.utils.path_utils import resolve_path

logger = logging.getLogger(__name__)


@dataclass
class ResolvedTrainingPaths:
    filtered_tiles_path: Path
    features_dir: Path
    targets_dir: Path
    models_dir: Path
    segmentation_dir: Optional[Path]
    slope_stripes_channel_dir: Optional[Path]
    path_key: str
    tile_size: int
    target_mode: str
    binary_threshold: float
    use_rgb: bool
    use_dem: bool
    use_slope: bool
    use_segmentation_layer: bool
    use_slope_stripes_channel: bool


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

    features_dir = resolve_path(Path(paths["features_dir"]), project_root)
    targets_dir = resolve_path(Path(paths["targets_dir"]), project_root)
    models_dir = resolve_path(Path(paths["models_dir"]), project_root)

    use_segmentation_layer = config["data"].get("use_segmentation_layer", False)
    segmentation_dir = None
    if use_segmentation_layer and paths.get("segmentation_dir"):
        segmentation_dir = resolve_path(Path(paths["segmentation_dir"]), project_root)

    use_slope_stripes_channel = config["data"].get("use_slope_stripes_channel", False)
    slope_stripes_channel_dir = None
    if use_slope_stripes_channel and paths.get("slope_stripes_channel_dir"):
        slope_stripes_channel_dir = resolve_path(
            Path(paths["slope_stripes_channel_dir"]), project_root
        )

    use_rgb = config["data"].get("use_rgb", True)
    use_dem = config["data"].get("use_dem", True)
    use_slope = config["data"].get("use_slope", True)

    if use_segmentation_layer:
        if not segmentation_dir or not segmentation_dir.exists():
            raise ValueError(
                "use_segmentation_layer is true but segmentation_dir is missing or does not exist. "
                "Add paths.<key>.segmentation_dir and tile the segmentation raster with the same tile size/overlap."
            )
    if use_slope_stripes_channel:
        if not slope_stripes_channel_dir or not slope_stripes_channel_dir.exists():
            raise ValueError(
                "use_slope_stripes_channel is true but slope_stripes_channel_dir is missing or does not exist. "
                "Add paths.<key>.slope_stripes_channel_dir and tile the slope-stripes raster with the same tile size/overlap."
            )

    return ResolvedTrainingPaths(
        filtered_tiles_path=filtered_tiles_path,
        features_dir=features_dir,
        targets_dir=targets_dir,
        models_dir=models_dir,
        segmentation_dir=segmentation_dir,
        slope_stripes_channel_dir=slope_stripes_channel_dir,
        path_key=path_key,
        tile_size=tile_size,
        target_mode=target_mode,
        binary_threshold=binary_threshold,
        use_rgb=use_rgb,
        use_dem=use_dem,
        use_slope=use_slope,
        use_segmentation_layer=use_segmentation_layer,
        use_slope_stripes_channel=use_slope_stripes_channel,
    )


def compute_in_channels(config_data: dict) -> int:
    use_rgb = config_data.get("use_rgb", True)
    use_dem = config_data.get("use_dem", True)
    use_slope = config_data.get("use_slope", True)
    use_segmentation_layer = config_data.get("use_segmentation_layer", False)
    use_slope_stripes_channel = config_data.get("use_slope_stripes_channel", False)
    n = (
        (3 if use_rgb else 0)
        + (1 if use_dem else 0)
        + (1 if use_slope else 0)
        + (1 if use_segmentation_layer else 0)
        + (1 if use_slope_stripes_channel else 0)
    )
    if n < 1:
        raise ValueError(
            "At least one input channel must be enabled. Set one or more of: "
            "use_rgb, use_dem, use_slope, use_segmentation_layer, use_slope_stripes_channel to true."
        )
    return n


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


def log_training_config_summary(resolved: ResolvedTrainingPaths, mode: str) -> None:
    r = resolved
    in_channels = compute_in_channels({
        "use_rgb": r.use_rgb,
        "use_dem": r.use_dem,
        "use_slope": r.use_slope,
        "use_segmentation_layer": r.use_segmentation_layer,
        "use_slope_stripes_channel": r.use_slope_stripes_channel,
    })
    logger.info("=== Training Configuration ===")
    logger.info(f"Mode: {mode} (tile size: {r.tile_size}x{r.tile_size}), target_mode: {r.target_mode}")
    logger.info(f"Filtered tiles: {r.filtered_tiles_path}")
    logger.info(f"Features dir: {r.features_dir}")
    logger.info(f"Targets dir: {r.targets_dir}")
    logger.info(f"Models dir: {r.models_dir}")
    if r.use_segmentation_layer and r.segmentation_dir:
        logger.info(f"Segmentation dir: {r.segmentation_dir}")
    if r.use_slope_stripes_channel and r.slope_stripes_channel_dir:
        logger.info(f"Slope-stripes channel dir: {r.slope_stripes_channel_dir}")
    logger.info(
        f"Input channels: rgb={r.use_rgb}, dem={r.use_dem}, slope={r.use_slope}, "
        f"segmentation={r.use_segmentation_layer}, slope_stripes={r.use_slope_stripes_channel} -> {in_channels} channels"
    )


def get_normalization_stats(
    train_tiles: List[dict],
    features_dir: Path,
    use_rgb: bool,
    use_dem: bool,
    use_slope: bool,
    use_bg_aug: bool,
    extended_path: Path,
):
    if not (use_rgb or use_dem or use_slope):
        return {}

    from src.preprocessing.normalization import compute_statistics

    logger.info("Computing normalization statistics...")
    extended_loaded = use_bg_aug and extended_path.exists()
    tiles_for_stats = (
        [t for t in train_tiles if t.get("role") == "lobe"]
        if extended_loaded
        else train_tiles
    )
    train_feature_paths = [features_dir / t["features_path"] for t in tiles_for_stats]
    stats = compute_statistics(train_feature_paths)
    dem_mean = stats.get("dem", {}).get("mean", "N/A")
    dem_std = stats.get("dem", {}).get("std", "N/A")
    slope_mean = stats.get("slope", {}).get("mean", "N/A")
    slope_std = stats.get("slope", {}).get("std", "N/A")
    dem_mean_str = f"{dem_mean:.2f}" if isinstance(dem_mean, (int, float)) else str(dem_mean)
    dem_std_str = f"{dem_std:.2f}" if isinstance(dem_std, (int, float)) else str(dem_std)
    slope_mean_str = f"{slope_mean:.2f}" if isinstance(slope_mean, (int, float)) else str(slope_mean)
    slope_std_str = f"{slope_std:.2f}" if isinstance(slope_std, (int, float)) else str(slope_std)
    logger.info(f"DEM stats: mean={dem_mean_str}, std={dem_std_str}")
    logger.info(f"Slope stats: mean={slope_mean_str}, std={slope_std_str}")
    return stats
