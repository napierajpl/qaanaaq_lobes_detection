"""Unit tests for synthetic_parenthesis mode (512-only) and dataset generation."""

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from src.utils.config_utils import get_training_path_key


class TestGetTrainingPathKey:
    def test_dev_256(self):
        assert get_training_path_key("dev", 256) == "dev"

    def test_dev_512(self):
        assert get_training_path_key("dev", 512) == "dev_512"

    def test_production_256(self):
        assert get_training_path_key("production", 256) == "production"

    def test_production_512(self):
        assert get_training_path_key("production", 512) == "production_512"

    def test_synthetic_parenthesis_uses_512_key(self):
        assert get_training_path_key("synthetic_parenthesis", 512) == "synthetic_parenthesis_512"

    def test_synthetic_parenthesis_256_still_returns_256_key(self):
        assert get_training_path_key("synthetic_parenthesis", 256) == "synthetic_parenthesis"


class TestGenerateSyntheticParenthesisDataset:
    @pytest.fixture
    def source_tile_dir(self, tmp_path):
        """Create a minimal 512x512 5-band source tile and filtered_tiles.json."""
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        tile_id = "tile_0000"
        h = w = 512
        transform = from_bounds(0, 0, 1, 1, w, h)
        profile = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": 5,
            "dtype": "float32",
            "transform": transform,
            "crs": "EPSG:4326",
        }
        feat_path = features_dir / f"{tile_id}.tif"
        data = np.zeros((5, h, w), dtype=np.float32)
        data[0:3] = 100
        data[3] = 0
        data[4] = 0
        with rasterio.open(feat_path, "w", **profile) as src:
            src.write(data)
        filtered_path = tmp_path / "filtered_tiles.json"
        with open(filtered_path, "w", encoding="utf-8") as f:
            json.dump({
                "tiles": [
                    {
                        "tile_id": tile_id,
                        "features_path": f"{tile_id}.tif",
                        "targets_path": f"{tile_id}.tif",
                    }
                ]
            }, f)
        return tmp_path

    def test_generate_dataset_creates_structure(
        self, source_tile_dir, tmp_path
    ):
        from scripts.generate_synthetic_parenthesis_dataset import generate_dataset

        out = tmp_path / "out"
        generate_dataset(
            source_filtered_tiles=source_tile_dir / "filtered_tiles.json",
            source_features_dir=source_tile_dir / "features",
            output_dir=out,
            tile_size=512,
            max_tiles=1,
            shapes_per_tile=1,
            shape_height_px=200,
            seed=42,
        )
        assert (out / "features").is_dir()
        assert (out / "targets").is_dir()
        assert (out / "filtered_tiles.json").exists()
        tiles_data = json.loads((out / "filtered_tiles.json").read_text())
        assert "tiles" in tiles_data
        assert len(tiles_data["tiles"]) == 1
        assert tiles_data.get("tile_size") == 512

    def test_generate_dataset_output_raster_shapes(
        self, source_tile_dir, tmp_path
    ):
        from scripts.generate_synthetic_parenthesis_dataset import generate_dataset

        out = tmp_path / "out"
        generate_dataset(
            source_filtered_tiles=source_tile_dir / "filtered_tiles.json",
            source_features_dir=source_tile_dir / "features",
            output_dir=out,
            tile_size=512,
            max_tiles=1,
            shapes_per_tile=1,
            shape_height_px=200,
            seed=42,
        )
        feat_path = out / "features" / "tile_0000.tif"
        tgt_path = out / "targets" / "tile_0000.tif"
        assert feat_path.exists()
        assert tgt_path.exists()
        with rasterio.open(feat_path) as src:
            feat = src.read()
        assert feat.shape == (5, 512, 512)
        with rasterio.open(tgt_path) as src:
            tgt = src.read(1)
        assert tgt.shape == (512, 512)
        assert 0 in np.unique(tgt)
        assert 20 in np.unique(tgt)
