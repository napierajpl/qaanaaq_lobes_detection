"""
Tile filtering utilities for data quality control.
"""
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

import numpy as np
import rasterio
from tqdm import tqdm


class TileFilter:
    """Filters tiles based on data quality criteria."""

    def __init__(
        self,
        min_rgb_coverage: float = 0.01,
        include_background_only: bool = True,
        min_target_coverage: Optional[float] = None,
    ):
        """
        Initialize tile filter.

        Args:
            min_rgb_coverage: Minimum fraction of RGB pixels that must be valid (non-nodata).
                              Default 0.01 (1%) - very lenient to catch truly empty tiles.
            include_background_only: If True, include tiles with no targets (all background).
                                     If False, exclude tiles with no positive target values.
            min_target_coverage: Minimum fraction of target pixels that must be positive.
                                 If None, only checks if any positive pixels exist.
                                 Only used if include_background_only=False.
        """
        self.min_rgb_coverage = min_rgb_coverage
        self.include_background_only = include_background_only
        self.min_target_coverage = min_target_coverage

    def check_rgb_tile(self, tile_path: Path) -> Tuple[bool, Dict]:
        """
        Check if RGB tile has valid data.

        Args:
            tile_path: Path to RGB tile (should have 3 bands)

        Returns:
            Tuple of (is_valid, stats_dict)
            stats_dict contains: valid_pixels, total_pixels, coverage_ratio
        """
        with rasterio.open(tile_path) as src:
            if src.count < 3:
                return False, {"error": "RGB tile should have at least 3 bands"}

            # Read RGB bands (bands 1, 2, 3)
            rgb_data = src.read([1, 2, 3])

            # Check nodata values
            nodata_values = [src.nodata] * 3 if src.nodata is not None else [None] * 3

            # Count valid pixels (not nodata in any band)
            valid_mask = np.ones(rgb_data.shape[1:], dtype=bool)
            for band_idx, nodata in enumerate(nodata_values):
                if nodata is not None:
                    valid_mask = valid_mask & (rgb_data[band_idx] != nodata)
                else:
                    # If no nodata specified, check if all bands are not zero
                    # (common pattern: empty areas are all zeros)
                    valid_mask = valid_mask & (rgb_data[band_idx] != 0)

            total_pixels = rgb_data.shape[1] * rgb_data.shape[2]
            valid_pixels = np.sum(valid_mask)
            coverage_ratio = valid_pixels / total_pixels if total_pixels > 0 else 0.0

            is_valid = coverage_ratio >= self.min_rgb_coverage

            stats = {
                "valid_pixels": int(valid_pixels),
                "total_pixels": int(total_pixels),
                "coverage_ratio": float(coverage_ratio),
            }

            return is_valid, stats

    def check_target_tile(
        self,
        tile_path: Path,
        compute_baselines: bool = True,
        lobe_threshold: float = 5.0
    ) -> Tuple[bool, Dict]:
        """
        Check if target tile has any positive values (lobes) and compute baseline statistics.

        Args:
            tile_path: Path to target tile (proximity map)
            compute_baselines: If True, compute per-tile baseline metrics
            lobe_threshold: Threshold for lobe pixels (default: 5.0)

        Returns:
            Tuple of (has_targets, stats_dict)
            stats_dict contains: positive_pixels, total_pixels, coverage_ratio, max_value,
            and optionally baseline_metrics if compute_baselines=True
        """
        with rasterio.open(tile_path) as src:
            target_data = src.read(1)
            flat_data = target_data.flatten()

            # Check for positive values (proximity map: 1-10 are positive, 0 is background)
            positive_mask = target_data > 0

            total_pixels = target_data.size
            positive_pixels = np.sum(positive_mask)
            coverage_ratio = positive_pixels / total_pixels if total_pixels > 0 else 0.0
            max_value = float(np.max(target_data))

            has_targets = positive_pixels > 0

            stats = {
                "positive_pixels": int(positive_pixels),
                "total_pixels": int(total_pixels),
                "coverage_ratio": float(coverage_ratio),
                "max_value": max_value,
            }

            # Compute per-tile baseline statistics
            if compute_baselines:
                baseline_stats = self._compute_tile_baselines(flat_data, lobe_threshold)
                stats["baseline_metrics"] = baseline_stats

            return has_targets, stats

    def _compute_tile_baselines(
        self,
        target_values: np.ndarray,
        lobe_threshold: float = 5.0
    ) -> Dict:
        """
        Compute baseline metrics for a single tile.

        Args:
            target_values: Flattened array of target values
            lobe_threshold: Threshold for lobe pixels

        Returns:
            Dictionary with baseline statistics
        """
        # Basic statistics
        mean_val = float(np.mean(target_values))
        median_val = float(np.median(target_values))
        std_val = float(np.std(target_values))
        min_val = float(np.min(target_values))
        max_val = float(np.max(target_values))

        # Class imbalance
        lobe_mask = target_values >= lobe_threshold
        background_mask = target_values < lobe_threshold

        lobe_pixels = int(np.sum(lobe_mask))
        background_pixels = int(np.sum(background_mask))
        lobe_fraction = float(lobe_pixels / len(target_values)) if len(target_values) > 0 else 0.0

        # Baseline MAE strategies
        mae_predict_zero = float(np.mean(np.abs(target_values - 0.0)))
        mae_predict_mean = float(np.mean(np.abs(target_values - mean_val)))
        mae_predict_median = float(np.mean(np.abs(target_values - median_val)))

        # Baseline RMSE strategies
        rmse_predict_zero = float(np.sqrt(np.mean((target_values - 0.0) ** 2)))
        rmse_predict_mean = float(np.sqrt(np.mean((target_values - mean_val) ** 2)))

        # Per-class baselines
        lobe_values = target_values[lobe_mask]
        background_values = target_values[background_mask]

        if len(lobe_values) > 0:
            lobe_mean = float(np.mean(lobe_values))
            mae_lobe_predict_zero = float(np.mean(np.abs(lobe_values - 0.0)))
            mae_lobe_predict_mean = float(np.mean(np.abs(lobe_values - lobe_mean)))
        else:
            lobe_mean = 0.0
            mae_lobe_predict_zero = 0.0
            mae_lobe_predict_mean = 0.0

        if len(background_values) > 0:
            mae_background_predict_zero = float(np.mean(np.abs(background_values - 0.0)))
        else:
            mae_background_predict_zero = 0.0

        # Weighted baseline (optimal per-class strategy)
        if len(target_values) > 0:
            weighted_mae = (
                mae_background_predict_zero * len(background_values) +
                mae_lobe_predict_mean * len(lobe_values)
            ) / len(target_values)
        else:
            weighted_mae = 0.0

        # Baseline IoU (predict 0 everywhere = no lobes predicted)
        target_binary = (target_values >= lobe_threshold).astype(float)
        pred_zero_binary = np.zeros_like(target_binary)
        intersection = np.sum(pred_zero_binary * target_binary)
        union = np.sum(pred_zero_binary) + np.sum(target_binary) - intersection
        iou_predict_zero = float(intersection / union) if union > 0 else (1.0 if intersection == 0 else 0.0)

        return {
            "statistics": {
                "mean": mean_val,
                "median": median_val,
                "std": std_val,
                "min": min_val,
                "max": max_val,
            },
            "class_imbalance": {
                "lobe_pixels": lobe_pixels,
                "background_pixels": background_pixels,
                "lobe_fraction": lobe_fraction,
                "lobe_threshold": lobe_threshold,
            },
            "baseline_mae": {
                "predict_zero": mae_predict_zero,
                "predict_mean": mae_predict_mean,
                "predict_median": mae_predict_median,
                "weighted_optimal": float(weighted_mae),
            },
            "baseline_rmse": {
                "predict_zero": rmse_predict_zero,
                "predict_mean": rmse_predict_mean,
            },
            "baseline_iou": {
                "predict_zero": iou_predict_zero,
            },
            "per_class_baselines": {
                "lobe": {
                    "mean": lobe_mean,
                    "mae_predict_zero": mae_lobe_predict_zero,
                    "mae_predict_mean": mae_lobe_predict_mean,
                },
                "background": {
                    "mae_predict_zero": mae_background_predict_zero,
                },
            },
        }

    def filter_tile_pairs(
        self,
        features_dir: Path,
        targets_dir: Path,
        output_file: Optional[Path] = None,
        compute_baselines: bool = True,
        lobe_threshold: float = 5.0,
    ) -> Tuple[List[Dict], Dict]:
        """
        Filter tile pairs based on RGB and target criteria.

        Args:
            features_dir: Directory containing feature tiles (RGB)
            targets_dir: Directory containing target tiles (proximity maps)
            output_file: Optional path to save filtered tile list as JSON

        Returns:
            List of dicts with tile info: {
                "tile_id": "tile_0000",
                "features_path": "...",
                "targets_path": "...",
                "rgb_valid": True,
                "has_targets": True,
                "rgb_stats": {...},
                "target_stats": {...}
            }
        """
        features_dir = Path(features_dir)
        targets_dir = Path(targets_dir)

        # Find all feature tiles
        feature_tiles = sorted(features_dir.glob("**/tile_*.tif"))

        valid_tiles = []
        stats_summary = {
            "total_tiles": 0,
            "rgb_invalid": 0,
            "background_only": 0,
            "valid_tiles": 0,
        }

        # Process tiles with progress bar
        pbar = tqdm(feature_tiles, desc="Filtering tiles", unit="tile")
        for feature_tile in pbar:
            stats_summary["total_tiles"] += 1

            # Find corresponding target tile
            tile_id = feature_tile.stem  # e.g., "tile_0000"
            target_tile = targets_dir / f"{tile_id}.tif"

            # Check if target tile exists
            if not target_tile.exists():
                # Try to find in subdirectories
                target_tile_list = list(targets_dir.glob(f"**/{tile_id}.tif"))
                if target_tile_list:
                    target_tile = target_tile_list[0]
                else:
                    pbar.set_postfix({
                        "valid": stats_summary["valid_tiles"],
                        "invalid": stats_summary["rgb_invalid"],
                        "bg_only": stats_summary["background_only"],
                    })
                    continue  # Skip if no matching target tile

            # Check RGB tile
            rgb_valid, rgb_stats = self.check_rgb_tile(feature_tile)
            if not rgb_valid:
                stats_summary["rgb_invalid"] += 1
                continue

            # Check target tile
            has_targets, target_stats = self.check_target_tile(
                target_tile,
                compute_baselines=compute_baselines,
                lobe_threshold=lobe_threshold
            )

            # Apply filtering logic
            if not self.include_background_only and not has_targets:
                stats_summary["background_only"] += 1
                pbar.set_postfix({
                    "valid": stats_summary["valid_tiles"],
                    "invalid": stats_summary["rgb_invalid"],
                    "bg_only": stats_summary["background_only"],
                })
                continue

            if self.min_target_coverage is not None:
                if target_stats["coverage_ratio"] < self.min_target_coverage:
                    stats_summary["background_only"] += 1
                    pbar.set_postfix({
                        "valid": stats_summary["valid_tiles"],
                        "invalid": stats_summary["rgb_invalid"],
                        "bg_only": stats_summary["background_only"],
                    })
                    continue

            # Tile passed all checks
            # Calculate relative paths from project root (assume features_dir is under data/processed/tiles/...)
            # Find common ancestor (project root)
            try:
                # Try relative to features_dir first
                features_rel = str(feature_tile.relative_to(features_dir))
                targets_rel = str(target_tile.relative_to(targets_dir))
            except ValueError:
                # If that fails, use absolute paths
                features_rel = str(feature_tile)
                targets_rel = str(target_tile)

            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                """Convert numpy types to native Python types."""
                if isinstance(obj, (np.integer, np.int_)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float_)):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_numpy_types(item) for item in obj]
                return obj

            valid_tiles.append({
                "tile_id": tile_id,
                "features_path": features_rel,
                "targets_path": targets_rel,
                "rgb_valid": bool(rgb_valid),
                "has_targets": bool(has_targets),
                "rgb_stats": convert_numpy_types(rgb_stats),
                "target_stats": convert_numpy_types(target_stats),
            })
            stats_summary["valid_tiles"] += 1

            # Update progress bar with current stats
            pbar.set_postfix({
                "valid": stats_summary["valid_tiles"],
                "invalid": stats_summary["rgb_invalid"],
                "bg_only": stats_summary["background_only"],
            })

        # Save to file if requested
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            tile_size = None
            if valid_tiles:
                first_feature = features_dir / valid_tiles[0]["features_path"]
                if first_feature.exists():
                    with rasterio.open(first_feature) as src:
                        tile_size = src.height if src.height == src.width else None

            output_data = {
                "filter_config": {
                    "min_rgb_coverage": self.min_rgb_coverage,
                    "include_background_only": self.include_background_only,
                    "min_target_coverage": self.min_target_coverage,
                },
                "tile_size": tile_size,
                "stats": stats_summary,
                "tiles": valid_tiles,
            }

            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2)

        return valid_tiles, stats_summary

    def print_summary(self, stats_summary: Dict):
        """Print filtering summary statistics."""
        total = stats_summary["total_tiles"]
        valid = stats_summary["valid_tiles"]
        rgb_invalid = stats_summary["rgb_invalid"]
        bg_only = stats_summary["background_only"]

        print("\n=== Tile Filtering Summary ===")
        print(f"Total tiles processed: {total}")
        print(f"  - RGB invalid (empty): {rgb_invalid} ({rgb_invalid/total*100:.1f}%)")
        print(f"  - Background only (excluded): {bg_only} ({bg_only/total*100:.1f}%)")
        print(f"  - Valid tiles: {valid} ({valid/total*100:.1f}%)")
        print("")
