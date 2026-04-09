"""Tests for the LayerRegistry."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from src.training.layer_registry import (
    LayerRegistry,
    LayerSpec,
    ResolvedLayer,
    build_layer_registry,
)


def _make_tile(tile_dir: Path, tile_id: str, bands: int = 1, value: float = 1.0):
    """Write a minimal GeoTIFF tile for testing."""
    path = tile_dir / f"{tile_id}.tif"
    tile_dir.mkdir(parents=True, exist_ok=True)
    h, w = 8, 8
    transform = from_bounds(0, 0, 1, 1, w, h)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w,
        count=bands, dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        for b in range(1, bands + 1):
            dst.write(np.full((h, w), value, dtype=np.float32), b)
    return path


def _spec(name, bands=1, norm="none", enabled=True, display=None):
    return LayerSpec(name=name, bands=bands, normalization=norm, enabled=enabled, display=display or {})


def _resolved(spec, tile_dir):
    return ResolvedLayer(spec=spec, tile_dir=tile_dir)


class TestLayerRegistryBasics:
    def test_in_channels_sums_enabled(self):
        layers = [
            _resolved(_spec("rgb", 3, enabled=True), Path(".")),
            _resolved(_spec("dem", 1, enabled=False), Path(".")),
            _resolved(_spec("slope", 1, enabled=True), Path(".")),
        ]
        reg = LayerRegistry(layers)
        assert reg.in_channels == 4

    def test_in_channels_raises_when_none_enabled(self):
        layers = [
            _resolved(_spec("rgb", 3, enabled=False), Path(".")),
        ]
        reg = LayerRegistry(layers)
        with pytest.raises(ValueError, match="At least one"):
            _ = reg.in_channels

    def test_layer_names(self):
        layers = [
            _resolved(_spec("rgb", 3, enabled=True), Path(".")),
            _resolved(_spec("dem", 1, enabled=False), Path(".")),
            _resolved(_spec("slope", 1, enabled=True), Path(".")),
        ]
        reg = LayerRegistry(layers)
        assert reg.layer_names() == ["rgb", "slope"]

    def test_has_layer(self):
        layers = [
            _resolved(_spec("rgb", 3, enabled=True), Path(".")),
            _resolved(_spec("dem", 1, enabled=False), Path(".")),
        ]
        reg = LayerRegistry(layers)
        assert reg.has_layer("rgb")
        assert not reg.has_layer("dem")

    def test_channel_range(self):
        layers = [
            _resolved(_spec("rgb", 3, enabled=True), Path(".")),
            _resolved(_spec("dem", 1, enabled=True), Path(".")),
            _resolved(_spec("slope", 1, enabled=True), Path(".")),
        ]
        reg = LayerRegistry(layers)
        assert reg.channel_range("rgb") == (0, 3)
        assert reg.channel_range("dem") == (3, 4)
        assert reg.channel_range("slope") == (4, 5)
        assert reg.channel_range("nonexistent") is None

    def test_channel_range_skips_disabled(self):
        layers = [
            _resolved(_spec("rgb", 3, enabled=True), Path(".")),
            _resolved(_spec("dem", 1, enabled=False), Path(".")),
            _resolved(_spec("slope", 1, enabled=True), Path(".")),
        ]
        reg = LayerRegistry(layers)
        assert reg.channel_range("slope") == (3, 4)
        assert reg.channel_range("dem") is None


class TestLayerRegistryLoading:
    def test_load_tile_concatenates_enabled_layers(self, tmp_path):
        rgb_dir = tmp_path / "rgb"
        dem_dir = tmp_path / "dem"
        _make_tile(rgb_dir, "tile_0000", bands=3, value=128.0)
        _make_tile(dem_dir, "tile_0000", bands=1, value=500.0)
        layers = [
            _resolved(_spec("rgb", 3, norm="rgb", enabled=True), rgb_dir),
            _resolved(_spec("dem", 1, norm="none", enabled=True), dem_dir),
        ]
        reg = LayerRegistry(layers)
        result = reg.load_tile("tile_0000", 8)
        assert result.shape == (4, 8, 8)
        np.testing.assert_allclose(result[0, 0, 0], 128.0 / 255.0, atol=1e-5)
        np.testing.assert_allclose(result[3, 0, 0], 500.0, atol=1e-5)

    def test_load_tile_skips_disabled(self, tmp_path):
        rgb_dir = tmp_path / "rgb"
        _make_tile(rgb_dir, "tile_0000", bands=3, value=100.0)
        layers = [
            _resolved(_spec("rgb", 3, norm="rgb", enabled=True), rgb_dir),
            _resolved(_spec("dem", 1, norm="none", enabled=False), Path(".")),
        ]
        reg = LayerRegistry(layers)
        result = reg.load_tile("tile_0000", 8)
        assert result.shape == (3, 8, 8)

    def test_load_tile_missing_file_raises(self, tmp_path):
        rgb_dir = tmp_path / "rgb"
        rgb_dir.mkdir()
        layers = [
            _resolved(_spec("rgb", 3, norm="rgb", enabled=True), rgb_dir),
        ]
        reg = LayerRegistry(layers)
        with pytest.raises(FileNotFoundError, match="rgb"):
            reg.load_tile("tile_9999", 8)

    def test_normalization_clip01(self, tmp_path):
        layer_dir = tmp_path / "ss"
        _make_tile(layer_dir, "tile_0000", bands=1, value=0.8)
        layers = [
            _resolved(_spec("ss", 1, norm="clip01", enabled=True), layer_dir),
        ]
        reg = LayerRegistry(layers)
        result = reg.load_tile("tile_0000", 8)
        assert result.shape == (1, 8, 8)
        np.testing.assert_allclose(result[0, 0, 0], 0.8, atol=1e-5)

    def test_normalization_standardize(self, tmp_path):
        layer_dir = tmp_path / "dem"
        _make_tile(layer_dir, "tile_0000", bands=1, value=500.0)
        layers = [
            _resolved(_spec("dem", 1, norm="standardize", enabled=True), layer_dir),
        ]
        reg = LayerRegistry(layers, normalization_stats={"dem": {"mean": 500.0, "std": 1.0}})
        result = reg.load_tile("tile_0000", 8)
        np.testing.assert_allclose(result[0, 0, 0], 0.0, atol=1e-5)


class TestLayerRegistryNormalizationStats:
    def test_compute_stats(self, tmp_path):
        dem_dir = tmp_path / "dem"
        _make_tile(dem_dir, "tile_0000", bands=1, value=100.0)
        _make_tile(dem_dir, "tile_0001", bands=1, value=200.0)
        layers = [
            _resolved(_spec("dem", 1, norm="standardize", enabled=True), dem_dir),
            _resolved(_spec("rgb", 3, norm="rgb", enabled=True), tmp_path / "rgb"),
        ]
        reg = LayerRegistry(layers)
        stats = reg.compute_normalization_stats(["tile_0000", "tile_0001"])
        assert "dem" in stats
        np.testing.assert_allclose(stats["dem"]["mean"], 150.0, atol=1e-3)
        assert "rgb" not in stats


class TestBuildLayerRegistry:
    def test_builds_from_config(self, tmp_path):
        rgb_dir = tmp_path / "rgb"
        rgb_dir.mkdir()
        _make_tile(rgb_dir, "tile_0000", bands=3, value=1.0)
        config = {
            "layers": {
                "rgb": {"bands": 3, "normalization": "rgb", "enabled": True},
                "dem": {"bands": 1, "normalization": "standardize", "enabled": False},
            },
            "paths": {
                "test_mode": {
                    "layer_dirs": {
                        "rgb": str(rgb_dir),
                    },
                },
            },
        }
        reg = build_layer_registry(config, tmp_path, "test_mode")
        assert reg.in_channels == 3
        assert reg.has_layer("rgb")
        assert not reg.has_layer("dem")

    def test_raises_when_enabled_layer_has_no_dir(self, tmp_path):
        config = {
            "layers": {
                "rgb": {"bands": 3, "normalization": "rgb", "enabled": True},
            },
            "paths": {
                "test_mode": {
                    "layer_dirs": {},
                },
            },
        }
        with pytest.raises(ValueError, match="no directory"):
            build_layer_registry(config, tmp_path, "test_mode")

    def test_raises_when_dir_does_not_exist(self, tmp_path):
        config = {
            "layers": {
                "rgb": {"bands": 3, "normalization": "rgb", "enabled": True},
            },
            "paths": {
                "test_mode": {
                    "layer_dirs": {
                        "rgb": str(tmp_path / "nonexistent"),
                    },
                },
            },
        }
        with pytest.raises(ValueError, match="does not exist"):
            build_layer_registry(config, tmp_path, "test_mode")
