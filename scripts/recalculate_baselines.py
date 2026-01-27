#!/usr/bin/env python3
"""Recalculate baseline metrics in filtered_tiles.json for updated proximity maps."""

import json
import sys
from pathlib import Path
from typing import Dict
import numpy as np
import rasterio
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.path_utils import get_project_root, resolve_path


def compute_tile_baselines(
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


def recalculate_baselines(
    filtered_tiles_path: Path,
    targets_dir: Path,
    lobe_threshold: float = 5.0,
) -> None:
    """
    Recalculate baseline metrics for all tiles in filtered_tiles.json.
    
    Args:
        filtered_tiles_path: Path to filtered_tiles.json
        targets_dir: Base directory containing target tiles
        lobe_threshold: Threshold for lobe pixels
    """
    print(f"Loading filtered tiles from: {filtered_tiles_path}")
    with open(filtered_tiles_path, 'r') as f:
        data = json.load(f)
    
    tiles = data.get("tiles", [])
    print(f"Found {len(tiles)} tiles to process")
    
    updated_count = 0
    error_count = 0
    
    for tile in tqdm(tiles, desc="Recalculating baselines"):
        try:
            # Get target tile path
            targets_path = tile.get("targets_path")
            if not targets_path:
                continue
            
            # Resolve full path
            target_file = targets_dir / targets_path
            
            if not target_file.exists():
                print(f"Warning: Target file not found: {target_file}")
                error_count += 1
                continue
            
            # Load target tile
            with rasterio.open(target_file) as src:
                target_data = src.read(1)
                flat_data = target_data.flatten()
            
            # Recalculate baseline metrics
            baseline_metrics = compute_tile_baselines(flat_data, lobe_threshold)
            
            # Update target_stats
            if "target_stats" not in tile:
                tile["target_stats"] = {}
            
            # Update statistics
            tile["target_stats"]["max_value"] = baseline_metrics["statistics"]["max"]
            
            # Update positive pixels and coverage (these might have changed with 20px)
            positive_mask = target_data > 0
            positive_pixels = int(np.sum(positive_mask))
            total_pixels = int(target_data.size)
            coverage_ratio = float(positive_pixels / total_pixels) if total_pixels > 0 else 0.0
            
            tile["target_stats"]["positive_pixels"] = positive_pixels
            tile["target_stats"]["total_pixels"] = total_pixels
            tile["target_stats"]["coverage_ratio"] = coverage_ratio
            
            # Update baseline metrics
            tile["target_stats"]["baseline_metrics"] = baseline_metrics
            
            updated_count += 1
            
        except Exception as e:
            print(f"Error processing tile {tile.get('tile_id', 'unknown')}: {e}")
            error_count += 1
            continue
    
    # Save updated JSON
    print("\nSaving updated filtered tiles...")
    with open(filtered_tiles_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print("\nCompleted:")
    print(f"  - Updated: {updated_count} tiles")
    print(f"  - Errors: {error_count} tiles")
    print(f"  - Updated file: {filtered_tiles_path}")


def main():
    """Recalculate baseline metrics."""
    import argparse
    
    project_root = get_project_root(__file__)
    
    parser = argparse.ArgumentParser(
        description="Recalculate baseline metrics in filtered_tiles.json"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data/processed/tiles/train/filtered_tiles.json",
        help="Path to filtered_tiles.json (default: data/processed/tiles/train/filtered_tiles.json)",
    )
    parser.add_argument(
        "--targets-dir",
        type=Path,
        default=project_root / "data/processed/tiles/train/targets",
        help="Base directory containing target tiles (default: data/processed/tiles/train/targets)",
    )
    parser.add_argument(
        "--lobe-threshold",
        type=float,
        default=5.0,
        help="Threshold for lobe pixels (default: 5.0)",
    )
    
    args = parser.parse_args()
    
    input_path = resolve_path(args.input, project_root)
    targets_dir = resolve_path(args.targets_dir, project_root)
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    if not targets_dir.exists():
        print(f"Error: Targets directory not found: {targets_dir}")
        sys.exit(1)
    
    recalculate_baselines(input_path, targets_dir, args.lobe_threshold)


if __name__ == "__main__":
    main()
