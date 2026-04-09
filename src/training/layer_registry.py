"""Layer registry: config-driven channel management for model inputs.

Each input layer (RGB, DEM, slope, slope_stripes, segmentation, etc.) is
defined in training_config.yaml under the ``layers`` key (ordered dict).
The registry resolves tile directories, loads and normalizes tiles, computes
channel counts, and provides metadata for visualization and augmentation.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio

from src.preprocessing.normalization import normalize_rgb, standardize_channel
from src.utils.path_utils import resolve_path

logger = logging.getLogger(__name__)


@dataclass
class LayerSpec:
    name: str
    bands: int
    normalization: str  # "rgb" | "standardize" | "segmentation" | "clip01" | "none"
    enabled: bool = True
    display: Dict = field(default_factory=dict)


@dataclass
class ResolvedLayer:
    spec: LayerSpec
    tile_dir: Path


class LayerRegistry:
    def __init__(
        self,
        layers: List[ResolvedLayer],
        normalization_stats: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self._all_layers = layers
        self._enabled = [l for l in layers if l.spec.enabled]
        self.normalization_stats = normalization_stats or {}

    @property
    def enabled_layers(self) -> List[ResolvedLayer]:
        return list(self._enabled)

    @property
    def all_layers(self) -> List[ResolvedLayer]:
        return list(self._all_layers)

    @property
    def in_channels(self) -> int:
        n = sum(l.spec.bands for l in self._enabled)
        if n < 1:
            raise ValueError(
                "At least one input layer must be enabled. "
                "Set at least one layer's enabled: true in the layers config."
            )
        return n

    def layer_names(self) -> List[str]:
        return [l.spec.name for l in self._enabled]

    def has_layer(self, name: str) -> bool:
        return any(l.spec.name == name for l in self._enabled)

    def get_layer(self, name: str) -> Optional[ResolvedLayer]:
        for l in self._enabled:
            if l.spec.name == name:
                return l
        return None

    def get_layer_dir(self, name: str) -> Optional[Path]:
        layer = self.get_layer(name)
        return layer.tile_dir if layer else None

    def channel_range(self, layer_name: str) -> Optional[Tuple[int, int]]:
        offset = 0
        for layer in self._enabled:
            if layer.spec.name == layer_name:
                return (offset, offset + layer.spec.bands)
            offset += layer.spec.bands
        return None

    # ── tile loading ────────────────────────────────────────────────

    def load_tile(self, tile_id: str, tile_size: int) -> np.ndarray:
        channels: List[np.ndarray] = []
        for layer in self._enabled:
            path = layer.tile_dir / f"{tile_id}.tif"
            if not path.exists():
                raise FileNotFoundError(
                    f"{layer.spec.name} tile not found: {path}"
                )
            data = self._load_and_normalize(path, layer)
            channels.append(data)
        if not channels:
            return np.zeros((1, tile_size, tile_size), dtype=np.float32)
        return np.concatenate(channels, axis=0)

    def _load_and_normalize(
        self, path: Path, layer: ResolvedLayer
    ) -> np.ndarray:
        norm = layer.spec.normalization
        if norm == "segmentation":
            return self._load_segmentation(path)
        with rasterio.open(path) as src:
            data = src.read()
        data = np.asarray(data, dtype=np.float32)
        if norm == "rgb":
            return normalize_rgb(data)
        if norm == "standardize":
            stats = self.normalization_stats.get(layer.spec.name, {})
            mean = stats.get("mean")
            std = stats.get("std")
            result, _, _ = standardize_channel(data, mean=mean, std=std)
            return result
        if norm == "clip01":
            return np.clip(data, 0.0, 1.0)
        if norm == "none":
            return data
        raise ValueError(
            f"Unknown normalization '{norm}' for layer '{layer.spec.name}'"
        )

    @staticmethod
    def _load_segmentation(
        path: Path, nodata: float = -9999.0
    ) -> np.ndarray:
        with rasterio.open(path) as src:
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

    # ── normalization statistics ────────────────────────────────────

    def compute_normalization_stats(
        self, train_tile_ids: List[str]
    ) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        for layer in self._enabled:
            if layer.spec.normalization != "standardize":
                continue
            values = []
            for tile_id in train_tile_ids:
                path = layer.tile_dir / f"{tile_id}.tif"
                if not path.exists():
                    continue
                with rasterio.open(path) as src:
                    data = src.read()
                values.append(data.flatten())
            if values:
                all_values = np.concatenate(values)
                stats[layer.spec.name] = {
                    "mean": float(np.mean(all_values)),
                    "std": float(np.std(all_values)),
                }
        self.normalization_stats = stats
        return stats


# ── factory ─────────────────────────────────────────────────────────


def build_layer_registry(
    config: dict,
    project_root: Path,
    path_key: str,
) -> LayerRegistry:
    layers_config = config.get("layers", {})
    if not layers_config:
        raise ValueError("No 'layers' defined in config.")

    paths_config = config.get("paths", {}).get(path_key, {})
    layer_dirs_config = paths_config.get("layer_dirs", {})

    resolved: List[ResolvedLayer] = []
    for name, lc in layers_config.items():
        spec = LayerSpec(
            name=name,
            bands=lc["bands"],
            normalization=lc.get("normalization", "none"),
            enabled=lc.get("enabled", True),
            display=lc.get("display", {}),
        )
        dir_path_str = layer_dirs_config.get(name)
        if spec.enabled and not dir_path_str:
            raise ValueError(
                f"Layer '{name}' is enabled but no directory specified in "
                f"paths.{path_key}.layer_dirs.{name}"
            )
        tile_dir = (
            resolve_path(Path(dir_path_str), project_root)
            if dir_path_str
            else Path(".")
        )
        if spec.enabled and not tile_dir.exists():
            raise ValueError(
                f"Layer '{name}' tile directory does not exist: {tile_dir}"
            )
        resolved.append(ResolvedLayer(spec=spec, tile_dir=tile_dir))

    return LayerRegistry(resolved)
