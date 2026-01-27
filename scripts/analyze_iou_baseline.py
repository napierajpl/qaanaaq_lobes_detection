#!/usr/bin/env python3
"""Analyze IoU baseline for random predictions with different proximity zones."""

import sys
from pathlib import Path
import numpy as np
import rasterio
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.path_utils import get_project_root


def analyze_random_baseline_iou(
    filtered_tiles_path: Path,
    targets_dir: Path,
    iou_threshold: float = 5.0,
    random_fraction: float = 0.1,  # Predict 10% of pixels randomly
) -> dict:
    """
    Calculate what IoU a random predictor would achieve.

    Args:
        filtered_tiles_path: Path to filtered_tiles.json
        targets_dir: Base directory containing target tiles
        iou_threshold: Threshold for IoU calculation
        random_fraction: Fraction of pixels to randomly predict as positive

    Returns:
        Dictionary with baseline statistics
    """
    import json

    with open(filtered_tiles_path, 'r') as f:
        data = json.load(f)

    tiles = data.get("tiles", [])

    total_intersection = 0
    total_pred_pixels = 0
    total_target_pixels = 0
    total_pixels = 0

    for tile in tqdm(tiles, desc="Analyzing random baseline"):
        targets_path = tile.get("targets_path")
        if not targets_path:
            continue

        target_file = targets_dir / targets_path
        if not target_file.exists():
            continue

        # Load target tile
        with rasterio.open(target_file) as src:
            target_data = src.read(1)

        # Binarize target
        target_binary = (target_data >= iou_threshold).astype(float)
        target_pixels = int(np.sum(target_binary))

        # Random prediction: predict random_fraction of pixels as positive
        tile_size = target_data.size
        num_random_predictions = int(tile_size * random_fraction)
        random_indices = np.random.choice(tile_size, size=num_random_predictions, replace=False)
        pred_binary = np.zeros_like(target_binary)
        pred_binary.flat[random_indices] = 1.0

        # Calculate intersection and union
        intersection = np.sum(pred_binary * target_binary)
        pred_pixels = int(np.sum(pred_binary))

        total_intersection += intersection
        total_pred_pixels += pred_pixels
        total_target_pixels += target_pixels
        total_pixels += tile_size

    # Calculate IoU
    total_union = total_pred_pixels + total_target_pixels - total_intersection
    random_iou = (total_intersection / total_union) if total_union > 0 else 0.0

    # Calculate expected IoU (theoretical)
    # If we predict random_fraction of pixels, and target_fraction are positive,
    # expected intersection = random_fraction * target_fraction * total_pixels
    target_fraction = total_target_pixels / total_pixels if total_pixels > 0 else 0
    expected_intersection = random_fraction * target_fraction * total_pixels
    expected_union = (random_fraction + target_fraction - random_fraction * target_fraction) * total_pixels
    expected_iou = (expected_intersection / expected_union) if expected_union > 0 else 0.0

    return {
        "random_iou": random_iou,
        "expected_iou": expected_iou,
        "target_fraction": target_fraction,
        "pred_fraction": random_fraction,
        "total_target_pixels": total_target_pixels,
        "total_pred_pixels": total_pred_pixels,
        "total_pixels": total_pixels,
        "intersection": total_intersection,
        "union": total_union,
    }


def compare_proximity_zones():
    """Compare random baseline IoU for 10px vs 20px proximity zones."""
    project_root = get_project_root(__file__)

    # We need to check if we have 10px tiles available
    # For now, let's analyze 20px and calculate what 10px would be theoretically

    filtered_tiles_path = project_root / "data/processed/tiles/train/filtered_tiles.json"
    targets_dir = project_root / "data/processed/tiles/train/targets"

    print("=" * 80)
    print("RANDOM BASELINE IoU ANALYSIS")
    print("=" * 80)
    print("\nAnalyzing 20px proximity maps...")

    # Analyze 20px
    results_20px = analyze_random_baseline_iou(
        filtered_tiles_path,
        targets_dir,
        iou_threshold=5.0,
        random_fraction=0.1,
    )

    print("\n" + "=" * 80)
    print("RESULTS (20px proximity, threshold=5.0)")
    print("=" * 80)
    print(f"Target fraction (pixels >= 5.0): {results_20px['target_fraction']:.4f} ({results_20px['target_fraction']*100:.2f}%)")
    print(f"Random prediction fraction: {results_20px['pred_fraction']:.4f} ({results_20px['pred_fraction']*100:.2f}%)")
    print(f"Random IoU (empirical): {results_20px['random_iou']:.6f}")
    print(f"Expected IoU (theoretical): {results_20px['expected_iou']:.6f}")
    print(f"\nTotal pixels: {results_20px['total_pixels']:,}")
    print(f"Target pixels (>= 5.0): {results_20px['total_target_pixels']:,}")
    print(f"Predicted pixels (random 10%): {results_20px['total_pred_pixels']:,}")
    print(f"Intersection: {results_20px['intersection']:,}")
    print(f"Union: {results_20px['union']:,}")

    # Theoretical calculation for 10px
    # If 20px has target_fraction, 10px would have roughly (10/20)^2 = 0.25x the area
    # But actually, it's more complex - pixels with value >= 5.0 in 10px would be different
    # Let's estimate: if 20px has X% pixels >= 5.0, 10px might have roughly X/4 % (area ratio)
    # But this is approximate

    print("\n" + "=" * 80)
    print("THEORETICAL COMPARISON")
    print("=" * 80)
    print("\nIf 10px proximity had 4x smaller target area:")
    estimated_10px_target_fraction = results_20px['target_fraction'] / 4
    estimated_10px_expected_iou = (
        (results_20px['pred_fraction'] * estimated_10px_target_fraction) /
        (results_20px['pred_fraction'] + estimated_10px_target_fraction -
         results_20px['pred_fraction'] * estimated_10px_target_fraction)
    ) if (results_20px['pred_fraction'] + estimated_10px_target_fraction) > 0 else 0.0

    print(f"Estimated 10px target fraction: {estimated_10px_target_fraction:.4f} ({estimated_10px_target_fraction*100:.2f}%)")
    print(f"Estimated 10px random IoU: {estimated_10px_expected_iou:.6f}")
    print(f"\n20px random IoU: {results_20px['expected_iou']:.6f}")
    print(f"Ratio (20px/10px): {results_20px['expected_iou'] / estimated_10px_expected_iou:.2f}x")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("\nIf random IoU is higher with 20px, then the task is indeed easier.")
    print("However, IoU should normalize for target size - the key question is:")
    print("  - Is the model actually learning better, or just predicting more pixels?")
    print("  - What fraction of pixels does the model predict >= threshold?")
    print("  - How does this compare between 10px and 20px runs?")


if __name__ == "__main__":
    compare_proximity_zones()
