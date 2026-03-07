"""
Data loading utilities for training.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Optional, Set

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
from torchvision.transforms.functional import adjust_contrast, adjust_saturation

from src.preprocessing.normalization import (
    normalize_rgb,
    standardize_dem,
    standardize_slope,
)


def get_all_tile_ids_from_dirs(
    features_dir: Path,
    targets_dir: Path,
) -> Set[str]:
    """Return set of tile_id that have both a feature file and a target file."""
    features_dir = Path(features_dir)
    targets_dir = Path(targets_dir)
    feature_stems = {f.stem for f in features_dir.glob("**/tile_*.tif")}
    target_stems = {f.stem for f in targets_dir.glob("**/tile_*.tif")}
    return feature_stems & target_stems


def _resolve_tile_path(base_dir: Path, tile_id: str) -> Optional[Path]:
    """Return path to tile_id.tif under base_dir (flat or nested)."""
    direct = base_dir / f"{tile_id}.tif"
    if direct.exists():
        return direct
    found = list(base_dir.glob(f"**/{tile_id}.tif"))
    return found[0] if found else None


def is_tile_rgb_all_white(
    features_path: Path,
    white_threshold: float = 0.95,
) -> bool:
    """
    Return True if at least white_threshold fraction of RGB pixels are white (all bands >= 250).
    """
    with rasterio.open(features_path) as src:
        if src.count < 3:
            return True
        rgb = src.read([1, 2, 3])
    rgb = np.transpose(rgb, (1, 2, 0))
    white = np.all(rgb >= 250, axis=-1)
    return float(np.mean(white)) >= white_threshold


def get_background_candidates(
    features_dir: Path,
    targets_dir: Path,
    valid_tile_ids: Set[str],
    white_threshold: float = 0.95,
    show_progress: bool = True,
) -> List[dict]:
    """
    Return list of tile dicts (tile_id, features_path, targets_path) for excluded tiles
    that are not all-white (suitable as background tiles).
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    features_dir = Path(features_dir)
    targets_dir = Path(targets_dir)
    all_ids = get_all_tile_ids_from_dirs(features_dir, targets_dir)
    excluded_ids = all_ids - valid_tile_ids
    n_excluded = len(excluded_ids)
    candidates = []
    it = sorted(excluded_ids)
    if show_progress and n_excluded:
        import sys
        print(f"  Checking {n_excluded} excluded tiles (open each to test non-white)...", flush=True)
    if show_progress and tqdm is not None:
        it = tqdm(it, desc="Scanning background candidates", unit="tile", total=n_excluded)
    for tile_id in it:
        feat_path = _resolve_tile_path(features_dir, tile_id)
        tgt_path = _resolve_tile_path(targets_dir, tile_id)
        if feat_path is None or tgt_path is None:
            continue
        if is_tile_rgb_all_white(feat_path, white_threshold):
            continue
        try:
            features_rel = str(feat_path.relative_to(features_dir))
            targets_rel = str(tgt_path.relative_to(targets_dir))
        except ValueError:
            features_rel = str(feat_path)
            targets_rel = str(tgt_path)
        candidates.append({
            "tile_id": tile_id,
            "features_path": features_rel,
            "targets_path": targets_rel,
        })
    return candidates


def build_extended_train_tiles(
    train_tiles: List[dict],
    background_candidates: List[dict],
    n_add: Optional[int] = None,
    random_seed: int = 42,
) -> List[dict]:
    """
    Build extended train list: train_tiles + n_add background + n_add augmented-lobe entries.
    n_add defaults to min(len(train_tiles), len(background_candidates)).
    Augmented entries have "augment": True and point to a lobe tile.
    Each entry gets "role": "lobe" | "background" | "augmented_lobe" for persistence.
    """
    rng = random.Random(random_seed)
    n_add = n_add or min(len(train_tiles), len(background_candidates))
    n_add = min(n_add, len(background_candidates))
    if n_add == 0:
        for t in train_tiles:
            t.setdefault("role", "lobe")
        return train_tiles

    for t in train_tiles:
        t.setdefault("role", "lobe")
    background_sample = [dict(t) for t in rng.sample(background_candidates, n_add)]
    for t in background_sample:
        t["role"] = "background"
    lobe_sample = rng.choices(train_tiles, k=n_add)
    augmented_entries = []
    for t in lobe_sample:
        entry = dict(t)
        entry["augment"] = True
        entry["role"] = "augmented_lobe"
        augmented_entries.append(entry)

    return train_tiles + background_sample + augmented_entries


def save_extended_training_tiles(
    output_path: Path,
    train_tiles: List[dict],
    config: Optional[dict] = None,
    stats: Optional[dict] = None,
) -> None:
    """Save extended training tile list (with role) to JSON for reuse and shapefile train_usage."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config or {},
        "stats": stats or {},
        "tiles": train_tiles,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    return None


def load_extended_training_tiles(path: Path) -> tuple:
    """Load extended_training_tiles.json; return (tiles, config, stats)."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return data.get("tiles", []), data.get("config", {}), data.get("stats", {})


def get_background_train_ids_from_extended_tiles(tiles: List[dict]) -> Set[str]:
    """Return set of tile_id where role == 'background'."""
    return {t["tile_id"] for t in tiles if t.get("role") == "background"}


def _apply_lobe_augmentation(
    features: torch.Tensor,
    target: torch.Tensor,
    contrast_range: Tuple[float, float] = (0.8, 1.2),
    saturation_range: Tuple[float, float] = (0.8, 1.2),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply random rotation (0/90/180/270) and random contrast/saturation to RGB only."""
    k = random.randint(0, 3)
    features = torch.rot90(features, k, dims=[1, 2])
    target = torch.rot90(target, k, dims=[1, 2])
    c = random.uniform(*contrast_range)
    s = random.uniform(*saturation_range)
    rgb = features[0:3].clone()
    rgb = adjust_contrast(rgb, c)
    rgb = adjust_saturation(rgb, s)
    features = features.clone()
    features[0:3] = rgb
    return features, target


def _load_and_normalize_segmentation_tile(
    segmentation_path: Path,
    tile_size: int,
    nodata: float = -9999.0,
) -> np.ndarray:
    """Load segmentation tile (1 band), replace nodata with 0, scale non-zero to (0, 1]. Shape (1, H, W)."""
    with rasterio.open(segmentation_path) as src:
        seg = src.read(1)
        nd = float(src.nodata if src.nodata is not None else nodata)
    seg = np.asarray(seg, dtype=np.float32)
    valid = seg != nd
    out = np.zeros_like(seg)
    if np.any(valid):
        v = seg[valid]
        v_max = float(np.max(v))
        if v_max > 0:
            out[valid] = v / v_max
    return out[np.newaxis, :, :]


class TileDataset(Dataset):
    """Dataset for loading feature and target tiles."""

    def __init__(
        self,
        tile_list: List[dict],
        features_base_dir: Path,
        targets_base_dir: Path,
        normalization_stats: Optional[dict] = None,
        tile_size: int = 256,
        augmentation_config: Optional[dict] = None,
        target_mode: str = "proximity",
        binary_threshold: float = 1.0,
        segmentation_base_dir: Optional[Path] = None,
        slope_stripes_base_dir: Optional[Path] = None,
        use_rgb: bool = True,
        use_dem: bool = True,
        use_slope: bool = True,
    ):
        self.tile_list = tile_list
        self.features_base_dir = Path(features_base_dir)
        self.targets_base_dir = Path(targets_base_dir)
        self.normalization_stats = normalization_stats or {}
        self.tile_size = tile_size
        self.augmentation_config = augmentation_config or {}
        self.target_mode = (target_mode or "proximity").lower()
        self.binary_threshold = binary_threshold
        self.segmentation_base_dir = Path(segmentation_base_dir) if segmentation_base_dir else None
        self.slope_stripes_base_dir = Path(slope_stripes_base_dir) if slope_stripes_base_dir else None
        self.use_rgb = use_rgb
        self.use_dem = use_dem
        self.use_slope = use_slope

    def __len__(self) -> int:
        return len(self.tile_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single tile pair.

        Returns:
            Tuple of (features, target) as tensors
        """
        tile_info = self.tile_list[idx]
        tile_id = tile_info.get("tile_id") or Path(tile_info["features_path"]).stem
        features_path_str = tile_info["features_path"].replace("\\", "/")
        features_path = Path(features_path_str) if Path(features_path_str).is_absolute() else self.features_base_dir / features_path_str

        channel_list: List[np.ndarray] = []

        if self.use_rgb or self.use_dem or self.use_slope:
            with rasterio.open(features_path) as src:
                all_bands = src.read()  # (5, H, W)
            h, w = all_bands.shape[1], all_bands.shape[2]
            if self.use_rgb:
                rgb = normalize_rgb(all_bands[0:3])
                channel_list.append(rgb)
            if self.use_dem:
                dem_mean = self.normalization_stats.get("dem", {}).get("mean")
                dem_std = self.normalization_stats.get("dem", {}).get("std")
                dem, _, _ = standardize_dem(all_bands[3:4], mean=dem_mean, std=dem_std)
                channel_list.append(dem)
            if self.use_slope:
                slope_mean = self.normalization_stats.get("slope", {}).get("mean")
                slope_std = self.normalization_stats.get("slope", {}).get("std")
                slope, _, _ = standardize_slope(all_bands[4:5], mean=slope_mean, std=slope_std)
                channel_list.append(slope)
        else:
            h, w = self.tile_size, self.tile_size

        if self.segmentation_base_dir is not None:
            seg_path = self.segmentation_base_dir / f"{tile_id}.tif"
            if not seg_path.exists():
                raise FileNotFoundError(f"Segmentation tile not found: {seg_path}")
            seg = _load_and_normalize_segmentation_tile(seg_path, self.tile_size)
            channel_list.append(seg)

        if self.slope_stripes_base_dir is not None:
            stripe_path = self.slope_stripes_base_dir / f"{tile_id}.tif"
            if not stripe_path.exists():
                raise FileNotFoundError(f"Slope-stripes tile not found: {stripe_path}")
            with rasterio.open(stripe_path) as src:
                stripe = src.read(1)
            stripe = np.clip(np.asarray(stripe, dtype=np.float32), 0.0, 1.0)[np.newaxis, :, :]
            if self.use_rgb or self.use_dem or self.use_slope:
                pass
            else:
                h, w = stripe.shape[1], stripe.shape[2]
            channel_list.append(stripe)

        features = np.concatenate(channel_list, axis=0) if channel_list else np.zeros((1, self.tile_size, self.tile_size), dtype=np.float32)

        targets_path_str = tile_info["targets_path"].replace("\\", "/")
        targets_path = Path(targets_path_str) if Path(targets_path_str).is_absolute() else self.targets_base_dir / targets_path_str
        with rasterio.open(targets_path) as src:
            target = src.read(1)

        assert features.shape[1] == self.tile_size and features.shape[2] == self.tile_size, \
            f"Feature tile size mismatch: {features.shape[1]}x{features.shape[2]}, expected {self.tile_size}x{self.tile_size}"
        assert target.shape[0] == self.tile_size and target.shape[1] == self.tile_size, \
            f"Target tile size mismatch: {target.shape}, expected {self.tile_size}x{self.tile_size}"

        # Convert to tensors
        features_tensor = torch.from_numpy(features).float()
        target_tensor = torch.from_numpy(target).float().unsqueeze(0)

        if self.target_mode == "binary":
            target_tensor = (target_tensor >= self.binary_threshold).float()

        if tile_info.get("augment"):
            features_tensor, target_tensor = _apply_lobe_augmentation(
                features_tensor,
                target_tensor,
                contrast_range=tuple(self.augmentation_config.get("contrast_range", (0.8, 1.2))),
                saturation_range=tuple(self.augmentation_config.get("saturation_range", (0.8, 1.2))),
            )

        return features_tensor, target_tensor


def load_filtered_tiles(filtered_tiles_path: Path, show_progress: bool = False) -> List[dict]:
    """
    Load filtered tile list from JSON.

    Args:
        filtered_tiles_path: Path to filtered_tiles.json
        show_progress: If True, show a progress bar while reading the file (chunked read).

    Returns:
        List of tile dictionaries
    """
    path = Path(filtered_tiles_path)
    if show_progress and path.exists():
        try:
            from tqdm import tqdm
            size = path.stat().st_size
            chunk_size = 512 * 1024
            chunks = []
            with open(path, "rb") as f:
                with tqdm(total=size, desc="Loading JSON", unit="B", unit_scale=True) as pbar:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        chunks.append(chunk)
                        pbar.update(len(chunk))
            data = json.loads(b"".join(chunks).decode("utf-8"))
            return data["tiles"]
        except Exception:
            pass
    with open(path) as f:
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
    tile_size: int = 256,
    augmentation_config: Optional[dict] = None,
    target_mode: str = "proximity",
    binary_threshold: float = 1.0,
    segmentation_base_dir: Optional[Path] = None,
    slope_stripes_base_dir: Optional[Path] = None,
    use_rgb: bool = True,
    use_dem: bool = True,
    use_slope: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = TileDataset(
        train_tiles,
        features_base_dir,
        targets_base_dir,
        normalization_stats,
        tile_size=tile_size,
        augmentation_config=augmentation_config,
        target_mode=target_mode,
        binary_threshold=binary_threshold,
        segmentation_base_dir=segmentation_base_dir,
        slope_stripes_base_dir=slope_stripes_base_dir,
        use_rgb=use_rgb,
        use_dem=use_dem,
        use_slope=use_slope,
    )

    val_dataset = TileDataset(
        val_tiles,
        features_base_dir,
        targets_base_dir,
        normalization_stats,
        tile_size=tile_size,
        target_mode=target_mode,
        binary_threshold=binary_threshold,
        segmentation_base_dir=segmentation_base_dir,
        slope_stripes_base_dir=slope_stripes_base_dir,
        use_rgb=use_rgb,
        use_dem=use_dem,
        use_slope=use_slope,
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
