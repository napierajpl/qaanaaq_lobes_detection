"""Data loading utilities for training."""

import json
import random
from pathlib import Path
from typing import List, Tuple, Optional, Set

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
from torchvision.transforms.functional import (
    adjust_contrast,
    adjust_gamma,
    adjust_hue,
    adjust_saturation,
)

from src.training.layer_registry import LayerRegistry


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
    return {t["tile_id"] for t in tiles if t.get("role") == "background"}


# ── augmentation ────────────────────────────────────────────────────


def _color_params_from_config(augmentation_config: dict) -> dict:
    return {
        "contrast_range": tuple(augmentation_config.get("contrast_range", (0.72, 1.28))),
        "saturation_range": tuple(augmentation_config.get("saturation_range", (0.72, 1.28))),
        "brightness_range": tuple(augmentation_config.get("brightness_range", (-0.12, 0.12))),
        "noise_std": float(augmentation_config.get("noise_std", 0.03)),
        "use_gamma": bool(augmentation_config.get("gamma", True)),
        "gamma_range": tuple(augmentation_config.get("gamma_range", (0.78, 1.22))),
        "use_hue": bool(augmentation_config.get("hue", True)),
        "hue_range": tuple(augmentation_config.get("hue_range", (-0.04, 0.04))),
    }


def _apply_lobe_augmentation(
    features: torch.Tensor,
    target: torch.Tensor,
    augmentation_config: dict,
    rgb_range: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    features, target = _apply_geometric_augmentation(features, target)
    color_kw = _color_params_from_config(augmentation_config)
    features = _apply_color_augmentation(features, rgb_range=rgb_range, **color_kw)
    return features, target


def _apply_geometric_augmentation(
    features: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    k = random.randint(0, 3)
    features = torch.rot90(features, k, dims=[1, 2])
    target = torch.rot90(target, k, dims=[1, 2])
    if random.random() > 0.5:
        features = torch.flip(features, dims=[2])
        target = torch.flip(target, dims=[2])
    if random.random() > 0.5:
        features = torch.flip(features, dims=[1])
        target = torch.flip(target, dims=[1])
    return features, target


def _apply_color_augmentation(
    features: torch.Tensor,
    rgb_range: Optional[Tuple[int, int]] = None,
    contrast_range: Tuple[float, float] = (0.72, 1.28),
    saturation_range: Tuple[float, float] = (0.72, 1.28),
    brightness_range: Tuple[float, float] = (-0.12, 0.12),
    noise_std: float = 0.03,
    use_gamma: bool = True,
    gamma_range: Tuple[float, float] = (0.78, 1.22),
    use_hue: bool = True,
    hue_range: Tuple[float, float] = (-0.04, 0.04),
) -> torch.Tensor:
    """Random photometric augmentation on RGB channels only."""
    if rgb_range is None:
        if features.shape[0] < 3:
            return features
        rgb_range = (0, 3)
    start, end = rgb_range
    if end - start < 3:
        return features
    features = features.clone()
    rgb = features[start:end]
    if use_hue:
        rgb = adjust_hue(rgb, random.uniform(*hue_range))
    rgb = adjust_saturation(rgb, random.uniform(*saturation_range))
    rgb = adjust_contrast(rgb, random.uniform(*contrast_range))
    if use_gamma:
        rgb = adjust_gamma(rgb, random.uniform(*gamma_range))
    rgb = rgb + random.uniform(*brightness_range)
    rgb = rgb + torch.randn_like(rgb) * noise_std
    features[start:end] = torch.clamp(rgb, 0.0, 1.0)
    return features


def _apply_train_augmentation(
    features: torch.Tensor,
    target: torch.Tensor,
    augmentation_config: dict,
    rgb_range: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if augmentation_config.get("geometric", True):
        features, target = _apply_geometric_augmentation(features, target)
    if augmentation_config.get("color", True):
        color_kw = _color_params_from_config(augmentation_config)
        features = _apply_color_augmentation(features, rgb_range=rgb_range, **color_kw)
    return features, target


# ── filtered tiles I/O ──────────────────────────────────────────────


def load_filtered_tiles(
    filtered_tiles_path: Path, show_progress: bool = False
) -> List[dict]:
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
        except ImportError:
            pass
    with open(path) as f:
        data = json.load(f)
    return data["tiles"]


def create_data_splits(
    tiles: List[dict],
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    split_sum = train_split + val_split + test_split
    if abs(split_sum - 1.0) >= 1e-6:
        raise ValueError(
            f"Data splits must sum to 1.0, got {split_sum:.6f}. "
            f"train={train_split}, val={val_split}, test={test_split}"
        )
    rng = random.Random(random_seed)
    shuffled = list(tiles)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


# ── dataset ─────────────────────────────────────────────────────────


class TileDataset(Dataset):
    """Dataset for loading feature and target tiles via LayerRegistry."""

    def __init__(
        self,
        tile_list: List[dict],
        targets_base_dir: Path,
        layer_registry: LayerRegistry,
        tile_size: int = 256,
        augmentation_config: Optional[dict] = None,
        target_mode: str = "proximity",
        binary_threshold: float = 1.0,
        train_augmentation: bool = False,
    ):
        self.tile_list = tile_list
        self.targets_base_dir = Path(targets_base_dir)
        self.layer_registry = layer_registry
        self.tile_size = tile_size
        self.augmentation_config = augmentation_config or {}
        self.target_mode = (target_mode or "proximity").lower()
        self.binary_threshold = binary_threshold
        self.train_augmentation = train_augmentation
        self._rgb_range = layer_registry.channel_range("rgb")

    def __len__(self) -> int:
        return len(self.tile_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tile_info = self.tile_list[idx]
        tile_id = tile_info.get("tile_id") or Path(tile_info.get("features_path", "")).stem

        features = self.layer_registry.load_tile(tile_id, self.tile_size)

        targets_path = tile_info.get("targets_path", f"{tile_id}.tif")
        targets_path_str = targets_path.replace("\\", "/")
        targets_full = (
            Path(targets_path_str)
            if Path(targets_path_str).is_absolute()
            else self.targets_base_dir / targets_path_str
        )
        with rasterio.open(targets_full) as src:
            target = src.read(1)

        assert features.shape[1] == self.tile_size and features.shape[2] == self.tile_size, \
            f"Feature tile size mismatch: {features.shape[1]}x{features.shape[2]}, expected {self.tile_size}x{self.tile_size}"
        assert target.shape[0] == self.tile_size and target.shape[1] == self.tile_size, \
            f"Target tile size mismatch: {target.shape}, expected {self.tile_size}x{self.tile_size}"

        features_tensor = torch.from_numpy(features).float()
        target_tensor = torch.from_numpy(target).float().unsqueeze(0)

        if self.target_mode == "binary":
            target_tensor = (target_tensor >= self.binary_threshold).float()

        if self.train_augmentation:
            features_tensor, target_tensor = _apply_train_augmentation(
                features_tensor, target_tensor, self.augmentation_config,
                rgb_range=self._rgb_range,
            )
        elif tile_info.get("augment"):
            features_tensor, target_tensor = _apply_lobe_augmentation(
                features_tensor, target_tensor, self.augmentation_config,
                rgb_range=self._rgb_range,
            )

        return features_tensor, target_tensor


# ── dataloader factory ──────────────────────────────────────────────


def create_dataloaders(
    train_tiles: List[dict],
    val_tiles: List[dict],
    targets_base_dir: Path,
    layer_registry: LayerRegistry,
    batch_size: int = 16,
    num_workers: int = 0,
    tile_size: int = 256,
    augmentation_config: Optional[dict] = None,
    target_mode: str = "proximity",
    binary_threshold: float = 1.0,
    train_augmentation: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = TileDataset(
        train_tiles,
        targets_base_dir,
        layer_registry,
        tile_size=tile_size,
        augmentation_config=augmentation_config,
        target_mode=target_mode,
        binary_threshold=binary_threshold,
        train_augmentation=train_augmentation,
    )

    val_dataset = TileDataset(
        val_tiles,
        targets_base_dir,
        layer_registry,
        tile_size=tile_size,
        target_mode=target_mode,
        binary_threshold=binary_threshold,
    )

    pin = torch.cuda.is_available()
    train_kw: dict = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin,
    }
    val_kw: dict = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin,
    }
    if num_workers > 0:
        train_kw["persistent_workers"] = True
        val_kw["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, **train_kw)
    val_loader = DataLoader(val_dataset, **val_kw)

    return train_loader, val_loader
