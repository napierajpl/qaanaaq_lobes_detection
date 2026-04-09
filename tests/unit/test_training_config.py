import logging
from pathlib import Path

import pytest

from src.training.training_config import (
    apply_illumination_filter,
    validate_data_splits,
    validate_in_channels,
)


class TestValidateInChannels:
    def test_all_layers_enabled(self):
        config = {
            "layers": {
                "rgb": {"bands": 3, "normalization": "rgb", "enabled": True},
                "dem": {"bands": 1, "normalization": "standardize", "enabled": True},
                "slope": {"bands": 1, "normalization": "standardize", "enabled": True},
                "segmentation": {"bands": 1, "normalization": "segmentation", "enabled": True},
                "slope_stripes": {"bands": 1, "normalization": "clip01", "enabled": True},
            },
            "model": {},
        }
        assert validate_in_channels(config) == 7

    def test_only_rgb(self):
        config = {
            "layers": {
                "rgb": {"bands": 3, "normalization": "rgb", "enabled": True},
                "dem": {"bands": 1, "normalization": "standardize", "enabled": False},
            },
            "model": {},
        }
        assert validate_in_channels(config) == 3

    def test_only_slope_stripes(self):
        config = {
            "layers": {
                "rgb": {"bands": 3, "normalization": "rgb", "enabled": False},
                "slope_stripes": {"bands": 1, "normalization": "clip01", "enabled": True},
            },
            "model": {},
        }
        assert validate_in_channels(config) == 1

    def test_none_enabled_raises(self):
        config = {
            "layers": {
                "rgb": {"bands": 3, "normalization": "rgb", "enabled": False},
                "dem": {"bands": 1, "normalization": "standardize", "enabled": False},
            },
            "model": {},
        }
        with pytest.raises(ValueError, match="At least one input layer"):
            validate_in_channels(config)

    def test_defaults_enabled_true_when_missing(self):
        config = {
            "layers": {
                "rgb": {"bands": 3, "normalization": "rgb"},
                "dem": {"bands": 1, "normalization": "standardize"},
            },
            "model": {},
        }
        assert validate_in_channels(config) == 4

    def test_matching_yaml_value_no_warning(self, caplog):
        config = {
            "layers": {
                "rgb": {"bands": 3, "normalization": "rgb", "enabled": True},
                "dem": {"bands": 1, "normalization": "standardize", "enabled": True},
                "slope": {"bands": 1, "normalization": "standardize", "enabled": True},
            },
            "model": {"in_channels": 5},
        }
        with caplog.at_level(logging.WARNING, logger="src.training.training_config"):
            result = validate_in_channels(config)
        assert result == 5
        assert "mismatch" not in caplog.text.lower()

    def test_mismatched_yaml_value_warns(self, caplog):
        config = {
            "layers": {
                "rgb": {"bands": 3, "normalization": "rgb", "enabled": True},
                "dem": {"bands": 1, "normalization": "standardize", "enabled": True},
                "slope": {"bands": 1, "normalization": "standardize", "enabled": True},
            },
            "model": {"in_channels": 3},
        }
        with caplog.at_level(logging.WARNING, logger="src.training.training_config"):
            result = validate_in_channels(config)
        assert result == 5
        assert "3" in caplog.text
        assert "5" in caplog.text


class TestValidateDataSplits:
    def test_valid_splits_pass(self):
        validate_data_splits(0.7, 0.15, 0.15)

    def test_exact_one_passes(self):
        validate_data_splits(1.0, 0.0, 0.0)

    def test_sum_too_high_raises(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            validate_data_splits(0.8, 0.2, 0.1)

    def test_sum_too_low_raises(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            validate_data_splits(0.5, 0.1, 0.1)


class TestApplyIlluminationFilter:
    @pytest.fixture
    def tiles(self):
        train = [
            {"id": "t1", "illumination": "sun"},
            {"id": "t2", "illumination": "shadow"},
            {"id": "t3", "illumination": "sun"},
            {"id": "t4", "role": "background"},
        ]
        val = [
            {"id": "v1", "illumination": "shadow"},
            {"id": "v2", "illumination": "sun"},
        ]
        test = [
            {"id": "e1", "illumination": "sun"},
        ]
        return train, val, test

    def test_filter_sun(self, tiles):
        train, val, test = tiles
        ft, fv, fe = apply_illumination_filter(
            train, val, test, "sun", illumination_include_background=False
        )
        assert [t["id"] for t in ft] == ["t1", "t3"]
        assert [t["id"] for t in fv] == ["v2"]
        assert [t["id"] for t in fe] == ["e1"]

    def test_filter_shadow(self, tiles):
        train, val, test = tiles
        ft, fv, fe = apply_illumination_filter(
            train, val, test, "shadow", illumination_include_background=False
        )
        assert [t["id"] for t in ft] == ["t2"]
        assert [t["id"] for t in fv] == ["v1"]
        assert fe == []

    def test_all_returns_unchanged(self, tiles):
        train, val, test = tiles
        ft, fv, fe = apply_illumination_filter(
            train, val, test, "all", illumination_include_background=False
        )
        assert ft is train
        assert fv is val
        assert fe is test

    def test_background_included_when_flag_set(self, tiles):
        train, val, test = tiles
        ft, fv, fe = apply_illumination_filter(
            train, val, test, "sun", illumination_include_background=True
        )
        assert [t["id"] for t in ft] == ["t1", "t3", "t4"]

    def test_background_excluded_when_flag_unset(self, tiles):
        train, val, test = tiles
        ft, fv, fe = apply_illumination_filter(
            train, val, test, "shadow", illumination_include_background=False
        )
        assert "t4" not in [t["id"] for t in ft]
