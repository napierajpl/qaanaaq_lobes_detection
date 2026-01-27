"""
Data normalization utilities.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rasterio


def normalize_rgb(rgb_data: np.ndarray) -> np.ndarray:
    """
    Normalize RGB data from [0, 255] to [0, 1].

    Args:
        rgb_data: RGB array with values in [0, 255]

    Returns:
        Normalized array with values in [0, 1]
    """
    return rgb_data.astype(np.float32) / 255.0


def standardize_dem(
    dem_data: np.ndarray,
    mean: Optional[float] = None,
    std: Optional[float] = None
) -> Tuple[np.ndarray, float, float]:
    """
    Standardize DEM data to mean=0, std=1.

    Args:
        dem_data: DEM array
        mean: Precomputed mean (if None, computed from data)
        std: Precomputed std (if None, computed from data)

    Returns:
        Tuple of (standardized array, mean, std)
    """
    if mean is None:
        mean = np.mean(dem_data)
    if std is None:
        std = np.std(dem_data)

    if std == 0:
        std = 1.0

    standardized = (dem_data.astype(np.float32) - mean) / std
    return standardized, mean, std


def standardize_slope(
    slope_data: np.ndarray,
    mean: Optional[float] = None,
    std: Optional[float] = None
) -> Tuple[np.ndarray, float, float]:
    """
    Standardize slope data to mean=0, std=1.

    Args:
        slope_data: Slope array
        mean: Precomputed mean (if None, computed from data)
        std: Precomputed std (if None, computed from data)

    Returns:
        Tuple of (standardized array, mean, std)
    """
    if mean is None:
        mean = np.mean(slope_data)
    if std is None:
        std = np.std(slope_data)

    if std == 0:
        std = 1.0

    standardized = (slope_data.astype(np.float32) - mean) / std
    return standardized, mean, std


def compute_statistics(tile_paths: List[Path]) -> Dict[str, Dict[str, float]]:
    """
    Compute normalization statistics from a set of tiles.

    Args:
        tile_paths: List of paths to feature tiles (5-band)

    Returns:
        Dictionary with statistics for DEM and slope
    """
    dem_values = []
    slope_values = []

    for tile_path in tile_paths:
        with rasterio.open(tile_path) as src:
            # Read all bands
            data = src.read()

            # Band 4 is DEM (0-indexed: band 4 = band 5 in 1-indexed)
            if src.count >= 4:
                dem_values.append(data[3].flatten())

            # Band 5 is slope (0-indexed: band 4 = band 5 in 1-indexed)
            if src.count >= 5:
                slope_values.append(data[4].flatten())

    stats = {}

    if dem_values:
        dem_all = np.concatenate(dem_values)
        stats["dem"] = {
            "mean": float(np.mean(dem_all)),
            "std": float(np.std(dem_all)),
        }

    if slope_values:
        slope_all = np.concatenate(slope_values)
        stats["slope"] = {
            "mean": float(np.mean(slope_all)),
            "std": float(np.std(slope_all)),
        }

    return stats
