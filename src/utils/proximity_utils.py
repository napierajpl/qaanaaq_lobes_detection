"""
Proximity map helpers: infer token and detect max value/distance from path or rasters.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def infer_proximity_token(targets_dir: str) -> str:
    """Infer proximity token from targets directory path (e.g. proximity10, proximity20)."""
    s = targets_dir.lower()
    if "proximity20" in s:
        return "proximity20"
    if "proximity10" in s:
        return "proximity10"
    return "unknown"


def detect_proximity_params(
    targets_dir: Path,
    val_tiles: Optional[List[dict]] = None,
    sample_size: int = 5,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Detect proximity max_value and max_distance from path or by sampling raster files.
    Returns (max_value, max_distance) or (None, None) if detection fails.
    """
    targets_path_str = str(targets_dir)
    if "proximity10px" in targets_path_str or "proximity10" in targets_path_str:
        return 10, 10
    if "proximity20px" in targets_path_str or "proximity20" in targets_path_str:
        return 20, 20
    if not val_tiles:
        return None, None
    try:
        import rasterio
    except ImportError:
        return None, None
    sample_tiles = val_tiles[: min(sample_size, len(val_tiles))]
    max_values: List[float] = []
    for tile_info in sample_tiles:
        tile_path = targets_dir / tile_info["targets_path"]
        if not tile_path.exists():
            continue
        try:
            with rasterio.open(tile_path) as raster_src:
                data = raster_src.read(1)
                max_values.append(float(data.max()))
        except (rasterio.RasterioIOError, ValueError, KeyError):
            continue
    if not max_values:
        return None, None
    detected_max = int(max(max_values))
    if detected_max <= 10:
        return 10, 10
    if detected_max <= 20:
        return 20, 20
    return detected_max, detected_max
