#!/usr/bin/env python3
"""Analyze IoU baseline for random predictions with different proximity zones."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.baseline_iou import analyze_random_baseline_iou
from src.utils.path_utils import get_project_root


def main():
    project_root = get_project_root(Path(__file__))
    filtered_tiles_path = project_root / "data/processed/tiles/train/filtered_tiles.json"
    targets_dir = project_root / "data/processed/tiles/train/targets"

    print("=" * 80)
    print("RANDOM BASELINE IoU ANALYSIS")
    print("=" * 80)
    print("\nAnalyzing 20px proximity maps...")

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

    estimated_10px_target_fraction = results_20px["target_fraction"] / 4
    denom = (
        results_20px["pred_fraction"] + estimated_10px_target_fraction
        - results_20px["pred_fraction"] * estimated_10px_target_fraction
    )
    estimated_10px_expected_iou = (
        (results_20px["pred_fraction"] * estimated_10px_target_fraction) / denom
        if denom > 0 else 0.0
    )

    print("\n" + "=" * 80)
    print("THEORETICAL COMPARISON")
    print("=" * 80)
    print("\nIf 10px proximity had 4x smaller target area:")
    print(f"Estimated 10px target fraction: {estimated_10px_target_fraction:.4f} ({estimated_10px_target_fraction*100:.2f}%)")
    print(f"Estimated 10px random IoU: {estimated_10px_expected_iou:.6f}")
    print(f"\n20px random IoU: {results_20px['expected_iou']:.6f}")
    if estimated_10px_expected_iou > 0:
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
    main()
