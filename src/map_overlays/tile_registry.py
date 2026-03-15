"""
Tile Registry: Single source of truth for tile metadata.

This module provides a TileRegistry class that manages all tile information
including geographic bounds, filtering status, splits, baseline metrics, and model metrics.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import rasterio
from rasterio.windows import Window

from src.data_processing.boundary_tile_filter import (
    load_boundary_union,
    tile_bounds_intersect_boundary,
)
from src.data_processing.tiling import Tiler
from src.training.dataloader import create_data_splits, load_filtered_tiles

logger = logging.getLogger(__name__)


class TileRegistry:
    """Registry for tile metadata including geographic bounds and metrics."""

    def __init__(self, registry_path: Path, source_raster_path: Optional[Path] = None):
        """
        Initialize tile registry.

        Args:
            registry_path: Path to registry JSON file
            source_raster_path: Path to source raster (for calculating bounds)
        """
        self.registry_path = Path(registry_path)
        self.source_raster_path = Path(source_raster_path) if source_raster_path else None
        self.registry: Dict = {}

        if self.registry_path.exists():
            self.load()
        else:
            self._initialize_empty()

    def _initialize_empty(self, tile_size: int = 256, overlap: float = 0.3) -> None:
        """Initialize empty registry structure."""
        self.registry = {
            "metadata": {
                "source_raster": str(self.source_raster_path) if self.source_raster_path else None,
                "tile_size": tile_size,
                "overlap": overlap,
                "crs": None,
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
            },
            "tiles": {},
        }

    def load(self) -> None:
        """Load registry from JSON file."""
        with open(self.registry_path, 'r') as f:
            self.registry = json.load(f)
        logger.info(f"Loaded tile registry from {self.registry_path}")

    def save(self) -> None:
        """Save registry to JSON file."""
        self.registry["metadata"]["last_updated"] = datetime.now().isoformat()
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
        logger.info(f"Saved tile registry to {self.registry_path}")

    def migrate_from_filtered_tiles(
        self,
        filtered_tiles_path: Path,
        source_raster_path: Path,
        features_dir: Path,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        random_seed: int = 42,
        tile_size: int = 256,
        overlap: float = 0.3,
    ) -> None:
        """
        Migrate data from filtered_tiles.json to registry format.

        Args:
            filtered_tiles_path: Path to filtered_tiles.json
            source_raster_path: Path to source raster for geographic bounds
            features_dir: Directory containing feature tiles
            train_split: Training split fraction
            val_split: Validation split fraction
            test_split: Test split fraction
            random_seed: Random seed for splits
        """
        logger.info("Migrating from filtered_tiles.json to tile registry...")

        # Load filtered tiles
        all_tiles = load_filtered_tiles(filtered_tiles_path)

        # Create splits (same logic as training)
        train_tiles, val_tiles, test_tiles = create_data_splits(
            all_tiles,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            random_seed=random_seed,
        )

        # Create split lookup
        split_lookup = {}
        for tile in train_tiles:
            split_lookup[tile["tile_id"]] = "train"
        for tile in val_tiles:
            split_lookup[tile["tile_id"]] = "val"
        for tile in test_tiles:
            split_lookup[tile["tile_id"]] = "test"

        # Load source raster for bounds calculation
        with rasterio.open(source_raster_path) as src:
            crs = src.crs.to_string() if src.crs else None
            transform = src.transform

            # Calculate tile grid
            tiler = Tiler(tile_size=tile_size, overlap=overlap)
            tile_grid = tiler.calculate_tile_grid(src.width, src.height)

            # Update metadata
            self.registry["metadata"]["source_raster"] = str(source_raster_path)
            self.registry["metadata"]["crs"] = crs
            self.registry["metadata"]["tile_size"] = tile_size
            self.registry["metadata"]["overlap"] = overlap

            # Process each tile
            valid_tile_ids = {tile["tile_id"] for tile in all_tiles}

            for tile_idx, (row_start, row_end, col_start, col_end) in enumerate(tile_grid):
                tile_id = f"tile_{tile_idx:04d}"

                # Calculate geographic bounds
                window = Window.from_slices((row_start, row_end), (col_start, col_end))
                tile_transform = rasterio.windows.transform(window, transform)

                # Get corner coordinates
                width = col_end - col_start
                height = row_end - row_start

                # Calculate bounds (minx, miny, maxx, maxy)
                # Transform pixel coordinates to geographic coordinates
                corners = [
                    rasterio.transform.xy(tile_transform, 0, 0),  # Top-left
                    rasterio.transform.xy(tile_transform, 0, width),  # Top-right
                    rasterio.transform.xy(tile_transform, height, 0),  # Bottom-left
                    rasterio.transform.xy(tile_transform, height, width),  # Bottom-right
                ]
                x_coords = [c[0] for c in corners]
                y_coords = [c[1] for c in corners]

                minx = min(x_coords)
                maxx = max(x_coords)
                miny = min(y_coords)
                maxy = max(y_coords)

                # Get tile info from filtered_tiles if available
                tile_info = next(
                    (t for t in all_tiles if t["tile_id"] == tile_id),
                    None
                )

                is_valid = tile_id in valid_tile_ids
                split = split_lookup.get(tile_id, None)

                # Build registry entry
                entry = {
                    "tile_id": tile_id,
                    "tile_idx": tile_idx,
                    "geographic_bounds": {
                        "minx": float(minx),
                        "miny": float(miny),
                        "maxx": float(maxx),
                        "maxy": float(maxy),
                    },
                    "pixel_bounds": {
                        "row_start": int(row_start),
                        "row_end": int(row_end),
                        "col_start": int(col_start),
                        "col_end": int(col_end),
                    },
                    "filtering": {
                        "is_valid": is_valid,
                        "rgb_valid": tile_info.get("rgb_valid", False) if tile_info else False,
                        "has_targets": tile_info.get("has_targets", False) if tile_info else False,
                    },
                    "split": split,
                }

                # Add paths if tile is valid
                if tile_info:
                    entry["paths"] = {
                        "features": tile_info.get("features_path", ""),
                        "targets": tile_info.get("targets_path", ""),
                    }
                    if "illumination" in tile_info:
                        entry["illumination"] = tile_info["illumination"]

                    # Add baseline metrics if available
                    if "target_stats" in tile_info and "baseline_metrics" in tile_info["target_stats"]:
                        entry["baseline_metrics"] = tile_info["target_stats"]["baseline_metrics"]

                self.registry["tiles"][tile_id] = entry

        logger.info(f"Migrated {len(self.registry['tiles'])} tiles to registry")
        self.save()

    def add_boundary_info(self, boundary_path: Path) -> None:
        """
        Set inside_boundary on each tile from geographic_bounds and boundary geometry.
        Call save() after if you want to persist.
        """
        boundary_path = Path(boundary_path)
        if not boundary_path.exists():
            logger.warning(f"Boundary file not found: {boundary_path}, skipping inside_boundary")
            return
        boundary_union = load_boundary_union(boundary_path)
        if boundary_union is None:
            logger.warning("Boundary is empty, setting inside_boundary to True for all")
        n = 0
        for tile_id, entry in self.registry["tiles"].items():
            bounds = entry.get("geographic_bounds")
            if not bounds:
                entry["inside_boundary"] = True
                n += 1
                continue
            entry["inside_boundary"] = tile_bounds_intersect_boundary(bounds, boundary_union)
            if entry["inside_boundary"]:
                n += 1
        logger.info(f"Boundary info added: {n}/{len(self.registry['tiles'])} tiles inside boundary")

    def update_model_metrics(
        self,
        run_id: str,
        tile_metrics: Dict[str, Dict[str, float]],
    ) -> None:
        """
        Update registry with model metrics from a training run.

        Args:
            run_id: MLflow run ID
            tile_metrics: Dictionary mapping tile_id to metrics dict
                Expected keys: mae, rmse, iou, improvement_over_baseline
        """
        logger.info(f"Updating registry with metrics from run {run_id}")

        updated_count = 0
        for tile_id, metrics in tile_metrics.items():
            if tile_id in self.registry["tiles"]:
                if "model_metrics" not in self.registry["tiles"][tile_id]:
                    self.registry["tiles"][tile_id]["model_metrics"] = {}

                self.registry["tiles"][tile_id]["model_metrics"][run_id] = {
                    "mae": float(metrics.get("mae", 0.0)),
                    "rmse": float(metrics.get("rmse", 0.0)),
                    "iou": float(metrics.get("iou", 0.0)),
                    "improvement_over_baseline": float(metrics.get("improvement_over_baseline", 0.0)),
                }
                updated_count += 1

        logger.info(f"Updated metrics for {updated_count} tiles")
        self.save()

    def get_tile(self, tile_id: str) -> Optional[Dict]:
        """Get tile entry by ID."""
        return self.registry["tiles"].get(tile_id)

    def get_all_tiles(self, filter_valid: bool = False, filter_split: Optional[str] = None) -> List[Dict]:
        """
        Get all tiles, optionally filtered.

        Args:
            filter_valid: If True, only return valid tiles
            filter_split: If provided, only return tiles from this split (train/val/test)

        Returns:
            List of tile entries
        """
        tiles = list(self.registry["tiles"].values())

        if filter_valid:
            tiles = [t for t in tiles if t.get("filtering", {}).get("is_valid", False)]

        if filter_split:
            tiles = [t for t in tiles if t.get("split") == filter_split]

        return tiles

    def get_metadata(self) -> Dict:
        """Get registry metadata."""
        return self.registry["metadata"]
