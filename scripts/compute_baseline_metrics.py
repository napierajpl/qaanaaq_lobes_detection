#!/usr/bin/env python3
"""
Compute baseline metrics for lobe detection task.

This script calculates what MAE, RMSE, and IoU we would get from naive baselines:
- Predicting 0 everywhere (most common value)
- Predicting mean proximity value everywhere
- Predicting median proximity value everywhere
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import rasterio
from src.training.dataloader import load_filtered_tiles
from src.utils.path_utils import get_project_root, resolve_path


def compute_baseline_metrics(
    filtered_tiles_path: Path,
    targets_dir: Path,
    split: str = "train",
) -> dict:
    """
    Compute baseline metrics for different naive strategies.
    
    Args:
        filtered_tiles_path: Path to filtered_tiles.json
        targets_dir: Directory containing target tiles
        split: Which split to analyze ('train', 'val', or 'all')
    
    Returns:
        Dictionary with baseline metrics
    """
    # Load tiles
    all_tiles = load_filtered_tiles(filtered_tiles_path)
    
    # Filter by split if needed
    if split == "train":
        # Use same split logic as training
        np.random.seed(42)
        indices = np.random.permutation(len(all_tiles))
        n_train = int(len(all_tiles) * 0.7)
        tiles = [all_tiles[i] for i in indices[:n_train]]
    elif split == "val":
        np.random.seed(42)
        indices = np.random.permutation(len(all_tiles))
        n_train = int(len(all_tiles) * 0.7)
        n_val = int(len(all_tiles) * 0.15)
        tiles = [all_tiles[i] for i in indices[n_train:n_train + n_val]]
    else:  # 'all'
        tiles = all_tiles
    
    print(f"Analyzing {len(tiles)} tiles from {split} split...")
    
    # Collect all target values
    all_values = []
    total_pixels = 0
    lobe_pixels = 0  # pixels with value >= 8
    
    for tile in tiles:
        target_path = targets_dir / tile["targets_path"]
        
        if not target_path.exists():
            print(f"Warning: {target_path} not found, skipping")
            continue
        
        with rasterio.open(target_path) as src:
            data = src.read(1)  # Read first band
            flat_data = data.flatten()
            
            all_values.extend(flat_data.tolist())
            total_pixels += len(flat_data)
            lobe_pixels += np.sum(data >= 8.0)
    
    all_values = np.array(all_values)
    
    # Compute statistics
    mean_val = np.mean(all_values)
    median_val = np.median(all_values)
    mode_val = 0.0  # Most common value (background)
    
    lobe_fraction = lobe_pixels / total_pixels if total_pixels > 0 else 0.0
    
    print("\n=== Target Statistics ===")
    print(f"Total pixels: {total_pixels:,}")
    print(f"Lobe pixels (>= 8.0): {lobe_pixels:,} ({lobe_fraction*100:.2f}%)")
    print(f"Mean proximity value: {mean_val:.4f}")
    print(f"Median proximity value: {median_val:.4f}")
    print(f"Mode proximity value: {mode_val:.4f}")
    print(f"Min value: {np.min(all_values):.4f}")
    print(f"Max value: {np.max(all_values):.4f}")
    print(f"Std deviation: {np.std(all_values):.4f}")
    
    # Compute baseline MAE and RMSE for each strategy
    baselines = {}
    
    # Strategy 1: Predict 0 everywhere
    mae_zero = np.mean(np.abs(all_values - 0.0))
    rmse_zero = np.sqrt(np.mean((all_values - 0.0) ** 2))
    baselines["predict_zero"] = {
        "mae": float(mae_zero),
        "rmse": float(rmse_zero),
        "description": "Predict 0 (background) for all pixels"
    }
    
    # Strategy 2: Predict mean everywhere
    mae_mean = np.mean(np.abs(all_values - mean_val))
    rmse_mean = np.sqrt(np.mean((all_values - mean_val) ** 2))
    baselines["predict_mean"] = {
        "mae": float(mae_mean),
        "rmse": float(rmse_mean),
        "description": f"Predict mean ({mean_val:.4f}) for all pixels"
    }
    
    # Strategy 3: Predict median everywhere
    mae_median = np.mean(np.abs(all_values - median_val))
    rmse_median = np.sqrt(np.mean((all_values - median_val) ** 2))
    baselines["predict_median"] = {
        "mae": float(mae_median),
        "rmse": float(rmse_median),
        "description": f"Predict median ({median_val:.4f}) for all pixels"
    }
    
    # Compute IoU for each baseline (using threshold >= 8.0)
    threshold = 8.0
    target_binary = (all_values >= threshold).astype(float)
    
    # Zero baseline: no predictions >= 8.0
    pred_zero_binary = np.zeros_like(target_binary)
    intersection_zero = np.sum(pred_zero_binary * target_binary)
    union_zero = np.sum(pred_zero_binary) + np.sum(target_binary) - intersection_zero
    iou_zero = (intersection_zero / union_zero) if union_zero > 0 else 0.0
    baselines["predict_zero"]["iou"] = float(iou_zero)
    
    # Mean baseline: predict mean >= 8.0?
    pred_mean_binary = (np.full_like(all_values, mean_val) >= threshold).astype(float)
    intersection_mean = np.sum(pred_mean_binary * target_binary)
    union_mean = np.sum(pred_mean_binary) + np.sum(target_binary) - intersection_mean
    iou_mean = (intersection_mean / union_mean) if union_mean > 0 else 0.0
    baselines["predict_mean"]["iou"] = float(iou_mean)
    
    # Median baseline: predict median >= 8.0?
    pred_median_binary = (np.full_like(all_values, median_val) >= threshold).astype(float)
    intersection_median = np.sum(pred_median_binary * target_binary)
    union_median = np.sum(pred_median_binary) + np.sum(target_binary) - intersection_median
    iou_median = (intersection_median / union_median) if union_median > 0 else 0.0
    baselines["predict_median"]["iou"] = float(iou_median)
    
    print("\n=== Baseline Metrics ===")
    for name, metrics in baselines.items():
        print(f"\n{metrics['description']}:")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  IoU:  {metrics['iou']:.4f}")
    
    # Additional analysis: separate metrics for lobe vs background pixels
    lobe_mask = all_values >= 8.0
    background_mask = all_values < 8.0
    
    lobe_values = all_values[lobe_mask]
    background_values = all_values[background_mask]
    
    print("\n=== Detailed Analysis ===")
    print("\nBackground pixels (value < 8.0):")
    print(f"  Count: {len(background_values):,} ({len(background_values)/len(all_values)*100:.2f}%)")
    print(f"  Mean: {np.mean(background_values):.4f}")
    mae_background = np.mean(np.abs(background_values - 0.0))
    print(f"  Baseline MAE (predict 0): {mae_background:.4f}")
    
    print("\nLobe pixels (value >= 8.0):")
    print(f"  Count: {len(lobe_values):,} ({len(lobe_values)/len(all_values)*100:.2f}%)")
    print(f"  Mean: {np.mean(lobe_values):.4f}")
    mae_lobe_zero = np.mean(np.abs(lobe_values - 0.0))
    mae_lobe_mean = np.mean(np.abs(lobe_values - np.mean(lobe_values)))
    print(f"  Baseline MAE (predict 0): {mae_lobe_zero:.4f}")
    print(f"  Baseline MAE (predict mean lobe value): {mae_lobe_mean:.4f}")
    
    # Weighted MAE: what if we compute MAE separately and weight by class frequency?
    weighted_mae = (mae_background * len(background_values) + mae_lobe_mean * len(lobe_values)) / len(all_values)
    
    print(f"\nWeighted baseline MAE (optimal per-class): {weighted_mae:.4f}")
    
    return {
        "statistics": {
            "total_pixels": int(total_pixels),
            "lobe_pixels": int(lobe_pixels),
            "lobe_fraction": float(lobe_fraction),
            "mean": float(mean_val),
            "median": float(median_val),
            "mode": float(mode_val),
            "min": float(np.min(all_values)),
            "max": float(np.max(all_values)),
            "std": float(np.std(all_values)),
        },
        "baselines": baselines,
        "detailed_analysis": {
            "background": {
                "count": int(len(background_values)),
                "fraction": float(len(background_values) / len(all_values)),
                "mean": float(np.mean(background_values)),
                "baseline_mae_predict_zero": float(mae_background),
            },
            "lobe": {
                "count": int(len(lobe_values)),
                "fraction": float(len(lobe_values) / len(all_values)),
                "mean": float(np.mean(lobe_values)),
                "baseline_mae_predict_zero": float(mae_lobe_zero),
                "baseline_mae_predict_mean": float(mae_lobe_mean),
            },
            "weighted_baseline_mae": float(weighted_mae),
        },
    }


def main():
    """Main function."""
    import argparse
    
    project_root = get_project_root(__file__)
    
    parser = argparse.ArgumentParser(
        description="Compute baseline metrics for lobe detection"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs" / "training_config.yaml",
        help="Path to training config file",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use dev tiles instead of production",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "all"],
        default="train",
        help="Which data split to analyze (default: train)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save baseline metrics JSON (optional)",
    )
    
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Determine paths
    mode = "dev" if args.dev else "production"
    paths = config["paths"][mode]
    
    filtered_tiles_path = resolve_path(Path(paths["filtered_tiles"]), project_root)
    targets_dir = resolve_path(Path(paths["targets_dir"]), project_root)
    
    print("=== Baseline Metrics Computation ===")
    print(f"Mode: {mode}")
    print(f"Split: {args.split}")
    print(f"Filtered tiles: {filtered_tiles_path}")
    print(f"Targets dir: {targets_dir}")
    print()
    
    # Compute baselines
    results = compute_baseline_metrics(
        filtered_tiles_path,
        targets_dir,
        split=args.split,
    )
    
    # Save results if requested
    if args.output:
        output_path = resolve_path(args.output, project_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n=== Summary ===")
    best_baseline_mae = min(b['mae'] for b in results['baselines'].values())
    print(f"Best baseline MAE: {best_baseline_mae:.4f}")
    print(f"Best baseline IoU: {max(b['iou'] for b in results['baselines'].values()):.4f}")
    
    if 'detailed_analysis' in results:
        weighted_mae = results['detailed_analysis']['weighted_baseline_mae']
        print(f"Weighted baseline MAE (optimal per-class): {weighted_mae:.4f}")
    
    print("\nYour model should beat these baselines!")
    print(f"\nNote: Your model's MAE of 0.3895 is {'BETTER' if 0.3895 < best_baseline_mae else 'WORSE'} than the naive baseline!")


if __name__ == "__main__":
    main()
