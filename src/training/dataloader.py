"""
Data loading utilities for training.
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio

from src.preprocessing.normalization import (
    normalize_rgb,
    standardize_dem,
    standardize_slope,
)


class TileDataset(Dataset):
    """Dataset for loading feature and target tiles."""

    def __init__(
        self,
        tile_list: List[dict],
        features_base_dir: Path,
        targets_base_dir: Path,
        normalization_stats: Optional[dict] = None,
    ):
        """
        Initialize dataset.

        Args:
            tile_list: List of tile dictionaries from filtered_tiles.json
            features_base_dir: Base directory for feature tiles
            targets_base_dir: Base directory for target tiles
            normalization_stats: Statistics for DEM/slope normalization
        """
        self.tile_list = tile_list
        self.features_base_dir = Path(features_base_dir)
        self.targets_base_dir = Path(targets_base_dir)
        self.normalization_stats = normalization_stats or {}

    def __len__(self) -> int:
        return len(self.tile_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single tile pair.

        Returns:
            Tuple of (features, target) as tensors
        """
        tile_info = self.tile_list[idx]

        # Load feature tile (5 bands: RGB + DEM + Slope)
        # Handle both absolute and relative paths (normalize Windows backslashes)
        features_path_str = tile_info["features_path"].replace("\\", "/")
        if Path(features_path_str).is_absolute():
            features_path = Path(features_path_str)
        else:
            features_path = self.features_base_dir / features_path_str

        with rasterio.open(features_path) as src:
            features = src.read()  # Shape: (5, H, W)

        # Load target tile (1 band: proximity map)
        targets_path_str = tile_info["targets_path"].replace("\\", "/")
        if Path(targets_path_str).is_absolute():
            targets_path = Path(targets_path_str)
        else:
            targets_path = self.targets_base_dir / targets_path_str

        with rasterio.open(targets_path) as src:
            target = src.read(1)  # Shape: (H, W)

        # Note: All tiles should be 256x256 due to tiling logic that adjusts
        # overlap for boundary tiles. If a tile is smaller, it's a bug in tiling.
        assert features.shape[1] == 256 and features.shape[2] == 256, \
            f"Feature tile size mismatch: {features.shape[1]}x{features.shape[2]}, expected 256x256"
        assert target.shape[0] == 256 and target.shape[1] == 256, \
            f"Target tile size mismatch: {target.shape}, expected 256x256"

        # Normalize features
        # RGB bands (0, 1, 2)
        features[0:3] = normalize_rgb(features[0:3])

        # DEM band (3)
        dem_mean = self.normalization_stats.get("dem", {}).get("mean")
        dem_std = self.normalization_stats.get("dem", {}).get("std")
        features[3], _, _ = standardize_dem(features[3], mean=dem_mean, std=dem_std)

        # Slope band (4)
        slope_mean = self.normalization_stats.get("slope", {}).get("mean")
        slope_std = self.normalization_stats.get("slope", {}).get("std")
        features[4], _, _ = standardize_slope(features[4], mean=slope_mean, std=slope_std)

        # Convert to tensors
        features_tensor = torch.from_numpy(features).float()
        target_tensor = torch.from_numpy(target).float().unsqueeze(0)  # Add channel dimension

        return features_tensor, target_tensor


def load_filtered_tiles(filtered_tiles_path: Path) -> List[dict]:
    """
    Load filtered tile list from JSON.

    Args:
        filtered_tiles_path: Path to filtered_tiles.json

    Returns:
        List of tile dictionaries
    """
    with open(filtered_tiles_path) as f:
        data = json.load(f)
    return data["tiles"]


def create_data_splits(
    tile_list: List[dict],
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Split tiles into train/val/test sets.

    Args:
        tile_list: List of all tiles
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_tiles, val_tiles, test_tiles)
    """
    # Validate splits sum to 1.0
    split_sum = train_split + val_split + test_split
    if abs(split_sum - 1.0) >= 1e-6:
        raise ValueError(
            f"Data splits must sum to 1.0, got {split_sum:.6f}. "
            f"train={train_split}, val={val_split}, test={test_split}"
        )

    np.random.seed(random_seed)
    indices = np.random.permutation(len(tile_list))

    n_train = int(len(tile_list) * train_split)
    n_val = int(len(tile_list) * val_split)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_tiles = [tile_list[i] for i in train_indices]
    val_tiles = [tile_list[i] for i in val_indices]
    test_tiles = [tile_list[i] for i in test_indices]

    return train_tiles, val_tiles, test_tiles


def create_dataloaders(
    train_tiles: List[dict],
    val_tiles: List[dict],
    features_base_dir: Path,
    targets_base_dir: Path,
    normalization_stats: dict,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        train_tiles: Training tile list
        val_tiles: Validation tile list
        features_base_dir: Base directory for feature tiles
        targets_base_dir: Base directory for target tiles
        normalization_stats: Statistics for normalization
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = TileDataset(
        train_tiles,
        features_base_dir,
        targets_base_dir,
        normalization_stats,
    )

    val_dataset = TileDataset(
        val_tiles,
        features_base_dir,
        targets_base_dir,
        normalization_stats,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader
