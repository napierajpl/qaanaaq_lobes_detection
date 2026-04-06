import pytest

from src.training.dataloader import (
    create_data_splits,
    build_extended_train_tiles,
    get_all_tile_ids_from_dirs,
    load_filtered_tiles,
    save_extended_training_tiles,
)


def _make_tiles(n):
    return [
        {
            "tile_id": f"tile_{i:04d}",
            "features_path": f"tile_{i:04d}.tif",
            "targets_path": f"tile_{i:04d}.tif",
        }
        for i in range(n)
    ]


# ── create_data_splits ──────────────────────────────────────────────


class TestCreateDataSplits:
    def test_correct_split_ratios(self):
        tiles = _make_tiles(100)
        train, val, test = create_data_splits(tiles, 0.7, 0.15, 0.15)
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_splits_sum_to_total(self):
        tiles = _make_tiles(97)
        train, val, test = create_data_splits(tiles, 0.7, 0.15, 0.15)
        assert len(train) + len(val) + len(test) == len(tiles)

    def test_no_tile_in_multiple_splits(self):
        tiles = _make_tiles(100)
        train, val, test = create_data_splits(tiles, 0.7, 0.15, 0.15)
        train_ids = {t["tile_id"] for t in train}
        val_ids = {t["tile_id"] for t in val}
        test_ids = {t["tile_id"] for t in test}
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_different_seeds_produce_different_splits(self):
        tiles = _make_tiles(100)
        train_a, _, _ = create_data_splits(tiles, 0.7, 0.15, 0.15, random_seed=1)
        train_b, _, _ = create_data_splits(tiles, 0.7, 0.15, 0.15, random_seed=2)
        ids_a = [t["tile_id"] for t in train_a]
        ids_b = [t["tile_id"] for t in train_b]
        assert ids_a != ids_b

    def test_same_seed_is_reproducible(self):
        tiles = _make_tiles(100)
        train_a, val_a, test_a = create_data_splits(tiles, 0.7, 0.15, 0.15, random_seed=42)
        train_b, val_b, test_b = create_data_splits(tiles, 0.7, 0.15, 0.15, random_seed=42)
        assert [t["tile_id"] for t in train_a] == [t["tile_id"] for t in train_b]
        assert [t["tile_id"] for t in val_a] == [t["tile_id"] for t in val_b]
        assert [t["tile_id"] for t in test_a] == [t["tile_id"] for t in test_b]

    def test_invalid_splits_raise_value_error(self):
        tiles = _make_tiles(10)
        with pytest.raises(ValueError, match="must sum to 1.0"):
            create_data_splits(tiles, 0.5, 0.3, 0.3)


# ── build_extended_train_tiles ───────────────────────────────────────


class TestBuildExtendedTrainTiles:
    def test_returns_train_tiles_when_no_background(self):
        train = _make_tiles(5)
        result = build_extended_train_tiles(train, background_candidates=[])
        assert len(result) == 5
        assert all(t["tile_id"].startswith("tile_") for t in result)

    def test_adds_background_and_augmented_entries(self):
        train = _make_tiles(5)
        bg = [
            {"tile_id": f"bg_{i:04d}", "features_path": f"bg_{i:04d}.tif", "targets_path": f"bg_{i:04d}.tif"}
            for i in range(10)
        ]
        result = build_extended_train_tiles(train, bg)
        n_add = min(len(train), len(bg))
        assert len(result) == len(train) + 2 * n_add

    def test_role_field_set_correctly(self):
        train = _make_tiles(3)
        bg = [
            {"tile_id": "bg_0000", "features_path": "bg_0000.tif", "targets_path": "bg_0000.tif"},
            {"tile_id": "bg_0001", "features_path": "bg_0001.tif", "targets_path": "bg_0001.tif"},
            {"tile_id": "bg_0002", "features_path": "bg_0002.tif", "targets_path": "bg_0002.tif"},
        ]
        result = build_extended_train_tiles(train, bg)
        roles = [t["role"] for t in result]
        assert roles.count("lobe") == 3
        assert roles.count("background") == 3
        assert roles.count("augmented_lobe") == 3

    def test_augment_flag_on_augmented_entries(self):
        train = _make_tiles(4)
        bg = [
            {"tile_id": f"bg_{i:04d}", "features_path": f"bg_{i:04d}.tif", "targets_path": f"bg_{i:04d}.tif"}
            for i in range(4)
        ]
        result = build_extended_train_tiles(train, bg)
        augmented = [t for t in result if t["role"] == "augmented_lobe"]
        assert all(t.get("augment") is True for t in augmented)
        non_augmented = [t for t in result if t["role"] != "augmented_lobe"]
        assert all(t.get("augment") is not True for t in non_augmented)


# ── get_all_tile_ids_from_dirs ───────────────────────────────────────


class TestGetAllTileIdsFromDirs:
    def test_returns_intersection(self, tmp_path):
        feat_dir = tmp_path / "features"
        tgt_dir = tmp_path / "targets"
        feat_dir.mkdir()
        tgt_dir.mkdir()
        for name in ["tile_0001.tif", "tile_0002.tif", "tile_0003.tif"]:
            (feat_dir / name).write_bytes(b"")
        for name in ["tile_0002.tif", "tile_0003.tif", "tile_0004.tif"]:
            (tgt_dir / name).write_bytes(b"")
        result = get_all_tile_ids_from_dirs(feat_dir, tgt_dir)
        assert result == {"tile_0002", "tile_0003"}

    def test_returns_empty_when_no_overlap(self, tmp_path):
        feat_dir = tmp_path / "features"
        tgt_dir = tmp_path / "targets"
        feat_dir.mkdir()
        tgt_dir.mkdir()
        (feat_dir / "tile_0001.tif").write_bytes(b"")
        (tgt_dir / "tile_0099.tif").write_bytes(b"")
        result = get_all_tile_ids_from_dirs(feat_dir, tgt_dir)
        assert result == set()


# ── save / load round-trip ───────────────────────────────────────────


class TestSaveLoadRoundTrip:
    def test_save_then_load_returns_same_tiles(self, tmp_path):
        tiles = _make_tiles(5)
        out_path = tmp_path / "extended_training_tiles.json"
        save_extended_training_tiles(out_path, tiles, config={"lr": 0.001}, stats={"n": 5})
        loaded = load_filtered_tiles(out_path)
        assert loaded == tiles
