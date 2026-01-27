#!/usr/bin/env python3
"""
Analyze per-tile model performance against baseline metrics.

This script helps identify:
- Which tiles the model is genuinely improving on vs just matching baseline
- Tiles with specific characteristics (high/low lobe ratio) that affect performance
- Whether improvements are real or just due to tile-specific properties
"""

import sys
import json
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import rasterio
import torch
from src.training.dataloader import load_filtered_tiles
from src.evaluation.metrics import compute_mae, compute_rmse, compute_iou
from src.utils.path_utils import get_project_root, resolve_path


def analyze_tile_performance(
    tile_info: Dict,
    targets_dir: Path,
    model_predictions: torch.Tensor,
    iou_threshold: float = 5.0,
) -> Dict:
    """
    Analyze model performance for a single tile and compare to baseline.

    Args:
        tile_info: Tile information from filtered_tiles.json (includes baseline_metrics)
        targets_dir: Directory containing target tiles
        model_predictions: Model predictions for this tile (torch.Tensor)
        iou_threshold: Threshold for IoU calculation

    Returns:
        Dictionary with performance metrics and comparison to baseline
    """
    # Load target tile
    target_path = targets_dir / tile_info["targets_path"]
    with rasterio.open(target_path) as src:
        target_data = src.read(1)

    # Convert to tensors
    target_tensor = torch.from_numpy(target_data).float()

    # Ensure predictions match target shape
    if model_predictions.shape != target_tensor.shape:
        # If predictions are batched or have channel dimension, squeeze
        model_predictions = model_predictions.squeeze()
        if model_predictions.shape != target_tensor.shape:
            raise ValueError(f"Shape mismatch: pred {model_predictions.shape} vs target {target_tensor.shape}")

    # Compute model metrics
    model_mae = compute_mae(model_predictions, target_tensor)
    model_rmse = compute_rmse(model_predictions, target_tensor)
    model_iou = compute_iou(model_predictions, target_tensor, threshold=iou_threshold)

    # Get baseline metrics from tile info
    baseline = tile_info.get("target_stats", {}).get("baseline_metrics", {})

    if not baseline:
        return {
            "tile_id": tile_info["tile_id"],
            "error": "No baseline metrics found for this tile",
            "model_mae": float(model_mae),
            "model_rmse": float(model_rmse),
            "model_iou": float(model_iou),
        }

    baseline_mae = baseline.get("baseline_mae", {})
    class_imbalance = baseline.get("class_imbalance", {})

    # Compare to baselines
    best_baseline_mae = baseline_mae.get("weighted_optimal", baseline_mae.get("predict_zero", float("inf")))
    baseline_mae_zero = baseline_mae.get("predict_zero", float("inf"))

    improvement_over_baseline = baseline_mae_zero - model_mae
    improvement_over_optimal = best_baseline_mae - model_mae

    # Determine if improvement is real
    is_better_than_baseline = model_mae < baseline_mae_zero
    is_better_than_optimal = model_mae < best_baseline_mae

    return {
        "tile_id": tile_info["tile_id"],
        "model_metrics": {
            "mae": float(model_mae),
            "rmse": float(model_rmse),
            "iou": float(model_iou),
        },
        "baseline_comparison": {
            "baseline_mae_zero": float(baseline_mae_zero),
            "baseline_mae_optimal": float(best_baseline_mae),
            "improvement_over_baseline": float(improvement_over_baseline),
            "improvement_over_optimal": float(improvement_over_optimal),
            "is_better_than_baseline": is_better_than_baseline,
            "is_better_than_optimal": is_better_than_optimal,
        },
        "tile_characteristics": {
            "lobe_fraction": float(class_imbalance.get("lobe_fraction", 0.0)),
            "lobe_pixels": int(class_imbalance.get("lobe_pixels", 0)),
            "background_pixels": int(class_imbalance.get("background_pixels", 0)),
        },
    }


def analyze_all_tiles(
    filtered_tiles_path: Path,
    targets_dir: Path,
    model_predictions_dict: Dict[str, torch.Tensor],
    iou_threshold: float = 5.0,
) -> Dict:
    """
    Analyze performance for all tiles and provide summary statistics.

    Args:
        filtered_tiles_path: Path to filtered_tiles.json
        targets_dir: Directory containing target tiles
        model_predictions_dict: Dictionary mapping tile_id to predictions tensor
        iou_threshold: Threshold for IoU calculation

    Returns:
        Dictionary with per-tile and aggregate analysis
    """
    all_tiles = load_filtered_tiles(filtered_tiles_path)

    per_tile_results = []
    tiles_better_than_baseline = 0
    tiles_better_than_optimal = 0

    for tile_info in all_tiles:
        tile_id = tile_info["tile_id"]

        if tile_id not in model_predictions_dict:
            print(f"Warning: No predictions found for {tile_id}, skipping")
            continue

        predictions = model_predictions_dict[tile_id]
        result = analyze_tile_performance(
            tile_info, targets_dir, predictions, iou_threshold
        )

        per_tile_results.append(result)

        if result.get("baseline_comparison", {}).get("is_better_than_baseline", False):
            tiles_better_than_baseline += 1

        if result.get("baseline_comparison", {}).get("is_better_than_optimal", False):
            tiles_better_than_optimal += 1

    # Aggregate statistics
    if per_tile_results:
        model_maes = [r["model_metrics"]["mae"] for r in per_tile_results if "model_metrics" in r]
        baseline_maes = [r["baseline_comparison"]["baseline_mae_zero"] for r in per_tile_results if "baseline_comparison" in r]
        improvements = [r["baseline_comparison"]["improvement_over_baseline"] for r in per_tile_results if "baseline_comparison" in r]
        lobe_fractions = [r["tile_characteristics"]["lobe_fraction"] for r in per_tile_results if "tile_characteristics" in r]

        aggregate_stats = {
            "total_tiles_analyzed": len(per_tile_results),
            "tiles_better_than_baseline": tiles_better_than_baseline,
            "tiles_better_than_optimal": tiles_better_than_optimal,
            "fraction_better_than_baseline": tiles_better_than_baseline / len(per_tile_results) if per_tile_results else 0.0,
            "fraction_better_than_optimal": tiles_better_than_optimal / len(per_tile_results) if per_tile_results else 0.0,
            "average_model_mae": float(np.mean(model_maes)) if model_maes else 0.0,
            "average_baseline_mae": float(np.mean(baseline_maes)) if baseline_maes else 0.0,
            "average_improvement": float(np.mean(improvements)) if improvements else 0.0,
            "average_lobe_fraction": float(np.mean(lobe_fractions)) if lobe_fractions else 0.0,
        }
    else:
        aggregate_stats = {}

    return {
        "per_tile_results": per_tile_results,
        "aggregate_stats": aggregate_stats,
    }


def main():
    """Main function - example usage."""
    import argparse

    project_root = get_project_root(__file__)

    parser = argparse.ArgumentParser(
        description="Analyze per-tile model performance against baselines"
    )
    parser.add_argument(
        "--filtered-tiles",
        type=Path,
        required=True,
        help="Path to filtered_tiles.json",
    )
    parser.add_argument(
        "--targets-dir",
        type=Path,
        required=True,
        help="Directory containing target tiles",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Directory containing model predictions (tile_id.tif files)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=5.0,
        help="IoU threshold (default: 5.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for analysis results (optional)",
    )

    args = parser.parse_args()

    # Resolve paths
    filtered_tiles_path = resolve_path(args.filtered_tiles, project_root)
    targets_dir = resolve_path(args.targets_dir, project_root)
    predictions_dir = resolve_path(args.predictions, project_root)

    # Load predictions
    print("Loading model predictions...")
    predictions_dict = {}
    for pred_file in predictions_dir.glob("tile_*.tif"):
        tile_id = pred_file.stem
        with rasterio.open(pred_file) as src:
            pred_data = src.read(1)
        predictions_dict[tile_id] = torch.from_numpy(pred_data).float()

    print(f"Loaded {len(predictions_dict)} predictions")

    # Analyze
    print("Analyzing per-tile performance...")
    results = analyze_all_tiles(
        filtered_tiles_path,
        targets_dir,
        predictions_dict,
        iou_threshold=args.iou_threshold,
    )

    # Print summary
    stats = results["aggregate_stats"]
    print("\n=== Per-Tile Performance Analysis ===")
    print(f"Total tiles analyzed: {stats.get('total_tiles_analyzed', 0)}")
    print(f"Tiles better than baseline: {stats.get('tiles_better_than_baseline', 0)} ({stats.get('fraction_better_than_baseline', 0)*100:.1f}%)")
    print(f"Tiles better than optimal: {stats.get('tiles_better_than_optimal', 0)} ({stats.get('fraction_better_than_optimal', 0)*100:.1f}%)")
    print(f"Average model MAE: {stats.get('average_model_mae', 0):.4f}")
    print(f"Average baseline MAE: {stats.get('average_baseline_mae', 0):.4f}")
    print(f"Average improvement: {stats.get('average_improvement', 0):.4f}")
    print(f"Average lobe fraction: {stats.get('average_lobe_fraction', 0)*100:.2f}%")

    # Save if requested
    if args.output:
        output_path = resolve_path(args.output, project_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
