"""Random baseline IoU analysis for proximity maps."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from tqdm import tqdm


def analyze_random_baseline_iou(
    filtered_tiles_path: Path,
    targets_dir: Path,
    iou_threshold: float = 5.0,
    random_fraction: float = 0.1,
) -> dict[str, Any]:
    """
    Calculate what IoU a random predictor would achieve (predict random_fraction of pixels as positive).
    """
    with open(filtered_tiles_path, encoding="utf-8") as f:
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

        with rasterio.open(target_file) as src:
            target_data = src.read(1)

        target_binary = (target_data >= iou_threshold).astype(float)
        target_pixels = int(np.sum(target_binary))

        tile_size = target_data.size
        num_random_predictions = int(tile_size * random_fraction)
        random_indices = np.random.choice(tile_size, size=num_random_predictions, replace=False)
        pred_binary = np.zeros_like(target_binary)
        pred_binary.flat[random_indices] = 1.0

        intersection = np.sum(pred_binary * target_binary)
        pred_pixels = int(np.sum(pred_binary))

        total_intersection += intersection
        total_pred_pixels += pred_pixels
        total_target_pixels += target_pixels
        total_pixels += tile_size

    total_union = total_pred_pixels + total_target_pixels - total_intersection
    random_iou = (total_intersection / total_union) if total_union > 0 else 0.0

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
