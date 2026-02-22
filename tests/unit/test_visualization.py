import pytest

from src.training.visualization import (
    _tile_id_to_index,
    resolve_representative_tiles,
)


class TestTileIdToIndex:
    def test_returns_index_when_last_segment_is_digit(self):
        assert _tile_id_to_index("features_tile_19189") == 19189
        assert _tile_id_to_index("rgb_tile_0000") == 0
        assert _tile_id_to_index("x_tile_19874") == 19874

    def test_returns_none_when_no_underscore(self):
        assert _tile_id_to_index("tile0000") is None

    def test_returns_none_when_last_segment_not_digit(self):
        assert _tile_id_to_index("features_tile_abc") is None
        assert _tile_id_to_index("tile_12_extra") is None


class TestResolveRepresentativeTiles:
    def test_match_by_int_index(self):
        all_tiles = [
            {"tile_id": "features_tile_0000", "features_path": "a", "targets_path": "b"},
            {"tile_id": "features_tile_19189", "features_path": "c", "targets_path": "d"},
            {"tile_id": "features_tile_19874", "features_path": "e", "targets_path": "f"},
        ]
        got = resolve_representative_tiles(all_tiles, [19189, 19874])
        assert len(got) == 2
        ids = {t["tile_id"] for t in got}
        assert ids == {"features_tile_19189", "features_tile_19874"}

    def test_match_by_str_tile_id(self):
        all_tiles = [
            {"tile_id": "features_tile_19189", "features_path": "c", "targets_path": "d"},
        ]
        got = resolve_representative_tiles(all_tiles, ["features_tile_19189"])
        assert len(got) == 1
        assert got[0]["tile_id"] == "features_tile_19189"

    def test_empty_config_ids_returns_empty(self):
        all_tiles = [{"tile_id": "features_tile_19189", "features_path": "c", "targets_path": "d"}]
        assert resolve_representative_tiles(all_tiles, []) == []

    def test_no_match_returns_empty(self):
        all_tiles = [{"tile_id": "features_tile_0000", "features_path": "a", "targets_path": "b"}]
        assert resolve_representative_tiles(all_tiles, [19189, 19874]) == []

    def test_mixed_int_and_str(self):
        all_tiles = [
            {"tile_id": "features_tile_19189", "features_path": "c", "targets_path": "d"},
            {"tile_id": "custom_tile_xyz", "features_path": "e", "targets_path": "f"},
        ]
        got = resolve_representative_tiles(all_tiles, [19189, "custom_tile_xyz"])
        assert len(got) == 2
        ids = {t["tile_id"] for t in got}
        assert ids == {"features_tile_19189", "custom_tile_xyz"}

    def test_tile_without_tile_id_skipped(self):
        all_tiles = [
            {"tile_id": "features_tile_19189", "features_path": "c", "targets_path": "d"},
            {"features_path": "e", "targets_path": "f"},
        ]
        got = resolve_representative_tiles(all_tiles, [19189])
        assert len(got) == 1
        assert got[0]["tile_id"] == "features_tile_19189"
