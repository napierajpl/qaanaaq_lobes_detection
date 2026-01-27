import pytest
import numpy as np
import rasterio
import json

from src.data_processing.tile_filter import TileFilter


@pytest.fixture
def valid_rgb_tile(tmp_path):
    """Create a valid RGB tile with data."""
    tile_path = tmp_path / "tile_0000.tif"
    
    height, width = 256, 256
    # Create RGB data with some variation (not all zeros)
    data = np.random.randint(10, 255, (3, height, width), dtype=np.uint8)
    
    transform = rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    
    with rasterio.open(
        tile_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3,
        dtype=data.dtype,
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        dst.write(data)
    
    return tile_path


@pytest.fixture
def empty_rgb_tile(tmp_path):
    """Create an empty RGB tile (all zeros or nodata)."""
    tile_path = tmp_path / "tile_0001.tif"
    
    height, width = 256, 256
    # All zeros (empty)
    data = np.zeros((3, height, width), dtype=np.uint8)
    
    transform = rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    
    with rasterio.open(
        tile_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3,
        dtype=data.dtype,
        crs='EPSG:4326',
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(data)
    
    return tile_path


@pytest.fixture
def sparse_rgb_tile(tmp_path):
    """Create RGB tile with very few valid pixels (< 1%)."""
    tile_path = tmp_path / "tile_0002.tif"
    
    height, width = 256, 256
    data = np.zeros((3, height, width), dtype=np.uint8)
    # Add just a few valid pixels (less than 1%)
    data[:, 0:5, 0:5] = 100
    
    transform = rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    
    with rasterio.open(
        tile_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3,
        dtype=data.dtype,
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        dst.write(data)
    
    return tile_path


@pytest.fixture
def target_tile_with_lobes(tmp_path):
    """Create target tile with lobe pixels (proximity map)."""
    tile_path = tmp_path / "target_tile_0000.tif"
    
    height, width = 256, 256
    # Create proximity map: mostly 0 (background), some positive values
    data = np.zeros((height, width), dtype=np.uint8)
    # Add some lobe pixels (value 10)
    data[100:110, 100:110] = 10
    # Add some proximity pixels (values 1-9)
    data[90:100, 100:110] = 5
    data[110:120, 100:110] = 3
    
    transform = rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    
    with rasterio.open(
        tile_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    
    return tile_path


@pytest.fixture
def target_tile_background_only(tmp_path):
    """Create target tile with only background (all zeros)."""
    tile_path = tmp_path / "target_tile_0001.tif"
    
    height, width = 256, 256
    data = np.zeros((height, width), dtype=np.uint8)
    
    transform = rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    
    with rasterio.open(
        tile_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    
    return tile_path


class TestTileFilterInitialization:
    """Test TileFilter initialization."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        filter_obj = TileFilter()
        assert filter_obj.min_rgb_coverage == 0.01
        assert filter_obj.include_background_only is True
        assert filter_obj.min_target_coverage is None
    
    def test_custom_initialization(self):
        """Test custom initialization."""
        filter_obj = TileFilter(
            min_rgb_coverage=0.05,
            include_background_only=False,
            min_target_coverage=0.02,
        )
        assert filter_obj.min_rgb_coverage == 0.05
        assert filter_obj.include_background_only is False
        assert filter_obj.min_target_coverage == 0.02


class TestCheckRGBTile:
    """Test RGB tile checking."""
    
    def test_valid_rgb_tile(self, valid_rgb_tile):
        """Test checking a valid RGB tile."""
        filter_obj = TileFilter(min_rgb_coverage=0.01)
        is_valid, stats = filter_obj.check_rgb_tile(valid_rgb_tile)
        
        assert is_valid == True
        assert stats["total_pixels"] == 256 * 256
        assert stats["coverage_ratio"] > 0.9  # Most pixels should be valid
        assert stats["valid_pixels"] > 0
    
    def test_empty_rgb_tile(self, empty_rgb_tile):
        """Test checking an empty RGB tile."""
        filter_obj = TileFilter(min_rgb_coverage=0.01)
        is_valid, stats = filter_obj.check_rgb_tile(empty_rgb_tile)
        
        assert is_valid == False
        assert stats["coverage_ratio"] == 0.0
        assert stats["valid_pixels"] == 0
    
    def test_sparse_rgb_tile_below_threshold(self, sparse_rgb_tile):
        """Test checking a sparse RGB tile below threshold."""
        filter_obj = TileFilter(min_rgb_coverage=0.01)
        is_valid, stats = filter_obj.check_rgb_tile(sparse_rgb_tile)
        
        # Should be invalid because coverage is very low (< 1%)
        assert is_valid == False
        assert stats["coverage_ratio"] < 0.01
    
    def test_sparse_rgb_tile_above_threshold(self, sparse_rgb_tile):
        """Test checking a sparse RGB tile with very low threshold."""
        filter_obj = TileFilter(min_rgb_coverage=0.0001)  # Very low threshold
        is_valid, stats = filter_obj.check_rgb_tile(sparse_rgb_tile)
        
        # Should be valid with very low threshold
        assert is_valid == True
        assert stats["coverage_ratio"] > 0.0
    
    def test_rgb_tile_insufficient_bands(self, tmp_path):
        """Test RGB tile with insufficient bands."""
        tile_path = tmp_path / "tile_2band.tif"
        
        height, width = 256, 256
        data = np.random.randint(0, 255, (2, height, width), dtype=np.uint8)
        
        with rasterio.open(
            tile_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=2,
            dtype=data.dtype,
            crs='EPSG:4326',
            transform=rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
        ) as dst:
            dst.write(data)
        
        filter_obj = TileFilter()
        is_valid, stats = filter_obj.check_rgb_tile(tile_path)
        
        assert is_valid is False
        assert "error" in stats


class TestCheckTargetTile:
    """Test target tile checking."""
    
    def test_target_tile_with_lobes(self, target_tile_with_lobes):
        """Test checking target tile with lobe pixels."""
        filter_obj = TileFilter()
        has_targets, stats = filter_obj.check_target_tile(target_tile_with_lobes)
        
        assert has_targets == True
        assert stats["positive_pixels"] > 0
        assert stats["coverage_ratio"] > 0.0
        assert stats["max_value"] == 10.0
    
    def test_target_tile_background_only(self, target_tile_background_only):
        """Test checking target tile with only background."""
        filter_obj = TileFilter()
        has_targets, stats = filter_obj.check_target_tile(target_tile_background_only)
        
        assert has_targets == False
        assert stats["positive_pixels"] == 0
        assert stats["coverage_ratio"] == 0.0
        assert stats["max_value"] == 0.0


class TestFilterTilePairs:
    """Test filtering tile pairs."""
    
    def test_filter_valid_pairs(self, tmp_path, valid_rgb_tile, target_tile_with_lobes):
        """Test filtering with valid RGB and target tiles."""
        features_dir = tmp_path / "features"
        targets_dir = tmp_path / "targets"
        features_dir.mkdir()
        targets_dir.mkdir()
        
        # Copy tiles to directories
        import shutil
        shutil.copy(valid_rgb_tile, features_dir / "tile_0000.tif")
        shutil.copy(target_tile_with_lobes, targets_dir / "tile_0000.tif")
        
        filter_obj = TileFilter(
            min_rgb_coverage=0.01,
            include_background_only=True,
        )
        
        valid_tiles, stats = filter_obj.filter_tile_pairs(
            features_dir=features_dir,
            targets_dir=targets_dir,
        )
        
        assert len(valid_tiles) == 1
        assert stats["total_tiles"] == 1
        assert stats["valid_tiles"] == 1
        assert stats["rgb_invalid"] == 0
        assert stats["background_only"] == 0
    
    def test_filter_exclude_empty_rgb(self, tmp_path, empty_rgb_tile, target_tile_with_lobes):
        """Test filtering excludes empty RGB tiles."""
        features_dir = tmp_path / "features"
        targets_dir = tmp_path / "targets"
        features_dir.mkdir()
        targets_dir.mkdir()
        
        import shutil
        shutil.copy(empty_rgb_tile, features_dir / "tile_0001.tif")
        shutil.copy(target_tile_with_lobes, targets_dir / "tile_0001.tif")
        
        filter_obj = TileFilter(min_rgb_coverage=0.01)
        
        valid_tiles, stats = filter_obj.filter_tile_pairs(
            features_dir=features_dir,
            targets_dir=targets_dir,
        )
        
        assert len(valid_tiles) == 0
        assert stats["rgb_invalid"] == 1
        assert stats["valid_tiles"] == 0
    
    def test_filter_exclude_background_only(self, tmp_path, valid_rgb_tile, target_tile_background_only):
        """Test filtering excludes background-only tiles when configured."""
        features_dir = tmp_path / "features"
        targets_dir = tmp_path / "targets"
        features_dir.mkdir()
        targets_dir.mkdir()
        
        import shutil
        shutil.copy(valid_rgb_tile, features_dir / "tile_0002.tif")
        shutil.copy(target_tile_background_only, targets_dir / "tile_0002.tif")
        
        filter_obj = TileFilter(
            min_rgb_coverage=0.01,
            include_background_only=False,  # Exclude background-only
        )
        
        valid_tiles, stats = filter_obj.filter_tile_pairs(
            features_dir=features_dir,
            targets_dir=targets_dir,
        )
        
        assert len(valid_tiles) == 0
        assert stats["background_only"] == 1
        assert stats["valid_tiles"] == 0
    
    def test_filter_include_background_only(self, tmp_path, valid_rgb_tile, target_tile_background_only):
        """Test filtering includes background-only tiles when configured."""
        features_dir = tmp_path / "features"
        targets_dir = tmp_path / "targets"
        features_dir.mkdir()
        targets_dir.mkdir()
        
        import shutil
        shutil.copy(valid_rgb_tile, features_dir / "tile_0003.tif")
        shutil.copy(target_tile_background_only, targets_dir / "tile_0003.tif")
        
        filter_obj = TileFilter(
            min_rgb_coverage=0.01,
            include_background_only=True,  # Include background-only
        )
        
        valid_tiles, stats = filter_obj.filter_tile_pairs(
            features_dir=features_dir,
            targets_dir=targets_dir,
        )
        
        assert len(valid_tiles) == 1
        assert stats["valid_tiles"] == 1
        assert stats["background_only"] == 0
    
    def test_filter_min_target_coverage(self, tmp_path, valid_rgb_tile):
        """Test filtering with minimum target coverage threshold."""
        features_dir = tmp_path / "features"
        targets_dir = tmp_path / "targets"
        features_dir.mkdir()
        targets_dir.mkdir()
        
        # Create target tile with very few positive pixels
        target_path = targets_dir / "tile_0004.tif"
        height, width = 256, 256
        data = np.zeros((height, width), dtype=np.uint8)
        data[0:5, 0:5] = 10  # Very few positive pixels (< 0.01%)
        
        with rasterio.open(
            target_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=data.dtype,
            crs='EPSG:4326',
            transform=rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
        ) as dst:
            dst.write(data, 1)
        
        import shutil
        shutil.copy(valid_rgb_tile, features_dir / "tile_0004.tif")
        
        filter_obj = TileFilter(
            min_rgb_coverage=0.01,
            include_background_only=False,
            min_target_coverage=0.01,  # Require at least 1% coverage
        )
        
        valid_tiles, stats = filter_obj.filter_tile_pairs(
            features_dir=features_dir,
            targets_dir=targets_dir,
        )
        
        # Should be excluded because target coverage is too low
        assert len(valid_tiles) == 0
        assert stats["background_only"] == 1
    
    def test_filter_saves_json(self, tmp_path, valid_rgb_tile, target_tile_with_lobes):
        """Test that filtering saves JSON output."""
        features_dir = tmp_path / "features"
        targets_dir = tmp_path / "targets"
        output_file = tmp_path / "filtered_tiles.json"
        features_dir.mkdir()
        targets_dir.mkdir()
        
        import shutil
        shutil.copy(valid_rgb_tile, features_dir / "tile_0005.tif")
        shutil.copy(target_tile_with_lobes, targets_dir / "tile_0005.tif")
        
        filter_obj = TileFilter()
        
        valid_tiles, stats = filter_obj.filter_tile_pairs(
            features_dir=features_dir,
            targets_dir=targets_dir,
            output_file=output_file,
        )
        
        assert output_file.exists()
        
        # Verify JSON is valid and contains expected data
        with open(output_file) as f:
            data = json.load(f)
        
        assert "filter_config" in data
        assert "stats" in data
        assert "tiles" in data
        assert len(data["tiles"]) == 1
        assert data["tiles"][0]["tile_id"] == "tile_0005"
        assert data["stats"]["valid_tiles"] == 1
    
    def test_filter_json_serialization_numpy_types(self, tmp_path, valid_rgb_tile, target_tile_with_lobes):
        """Test that JSON serialization handles numpy types correctly."""
        features_dir = tmp_path / "features"
        targets_dir = tmp_path / "targets"
        output_file = tmp_path / "filtered_tiles.json"
        features_dir.mkdir()
        targets_dir.mkdir()
        
        import shutil
        shutil.copy(valid_rgb_tile, features_dir / "tile_0006.tif")
        shutil.copy(target_tile_with_lobes, targets_dir / "tile_0006.tif")
        
        filter_obj = TileFilter()
        
        valid_tiles, stats = filter_obj.filter_tile_pairs(
            features_dir=features_dir,
            targets_dir=targets_dir,
            output_file=output_file,
        )
        
        # Should not raise TypeError about numpy types
        with open(output_file) as f:
            data = json.load(f)
        
        # Verify all values are native Python types
        tile = data["tiles"][0]
        assert isinstance(tile["rgb_valid"], bool)
        assert isinstance(tile["has_targets"], bool)
        assert isinstance(tile["rgb_stats"]["valid_pixels"], int)
        assert isinstance(tile["rgb_stats"]["coverage_ratio"], float)
        assert isinstance(tile["target_stats"]["max_value"], float)
    
    def test_filter_missing_target_tile(self, tmp_path, valid_rgb_tile):
        """Test filtering when target tile is missing."""
        features_dir = tmp_path / "features"
        targets_dir = tmp_path / "targets"
        features_dir.mkdir()
        targets_dir.mkdir()
        
        import shutil
        shutil.copy(valid_rgb_tile, features_dir / "tile_0007.tif")
        # Don't create corresponding target tile
        
        filter_obj = TileFilter()
        
        valid_tiles, stats = filter_obj.filter_tile_pairs(
            features_dir=features_dir,
            targets_dir=targets_dir,
        )
        
        # Should skip tiles without matching targets
        assert len(valid_tiles) == 0
        assert stats["total_tiles"] == 1
        assert stats["valid_tiles"] == 0
    
    def test_filter_multiple_tiles(self, tmp_path, valid_rgb_tile, empty_rgb_tile, 
                                   target_tile_with_lobes, target_tile_background_only):
        """Test filtering multiple tiles with mixed validity."""
        features_dir = tmp_path / "features"
        targets_dir = tmp_path / "targets"
        features_dir.mkdir()
        targets_dir.mkdir()
        
        import shutil
        # Valid RGB + valid target
        shutil.copy(valid_rgb_tile, features_dir / "tile_0008.tif")
        shutil.copy(target_tile_with_lobes, targets_dir / "tile_0008.tif")
        
        # Empty RGB + valid target (should be excluded)
        shutil.copy(empty_rgb_tile, features_dir / "tile_0009.tif")
        shutil.copy(target_tile_with_lobes, targets_dir / "tile_0009.tif")
        
        # Valid RGB + background only (should be excluded if configured)
        shutil.copy(valid_rgb_tile, features_dir / "tile_0010.tif")
        shutil.copy(target_tile_background_only, targets_dir / "tile_0010.tif")
        
        filter_obj = TileFilter(
            min_rgb_coverage=0.01,
            include_background_only=False,
        )
        
        valid_tiles, stats = filter_obj.filter_tile_pairs(
            features_dir=features_dir,
            targets_dir=targets_dir,
        )
        
        assert stats["total_tiles"] == 3
        assert stats["rgb_invalid"] == 1
        assert stats["background_only"] == 1
        assert stats["valid_tiles"] == 1
        assert len(valid_tiles) == 1
        assert valid_tiles[0]["tile_id"] == "tile_0008"


class TestPrintSummary:
    """Test summary printing."""
    
    def test_print_summary(self, capsys):
        """Test that summary prints correctly."""
        filter_obj = TileFilter()
        stats = {
            "total_tiles": 100,
            "rgb_invalid": 5,
            "background_only": 10,
            "valid_tiles": 85,
        }
        
        filter_obj.print_summary(stats)
        
        captured = capsys.readouterr()
        assert "Tile Filtering Summary" in captured.out
        assert "100" in captured.out
        assert "5" in captured.out
        assert "10" in captured.out
        assert "85" in captured.out
