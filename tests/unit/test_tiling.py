import pytest
import numpy as np
import rasterio

from src.data_processing.tiling import Tiler


@pytest.fixture
def sample_raster(tmp_path):
    """Create a sample test raster."""
    raster_path = tmp_path / "test_raster.tif"

    height, width = 1024, 1024
    data = np.random.randint(0, 255, (3, height, width), dtype=np.uint8)

    transform = rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)

    with rasterio.open(
        raster_path,
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

    return raster_path


def test_tiler_initialization():
    """Test Tiler initialization."""
    tiler = Tiler(tile_size=256, overlap=0.3)
    assert tiler.tile_size == 256
    assert tiler.overlap == 0.3
    assert tiler.stride == int(256 * 0.7)


def test_calculate_tile_grid():
    """Test tile grid calculation."""
    tiler = Tiler(tile_size=256, overlap=0.3)
    tiles = tiler.calculate_tile_grid(1024, 1024)

    assert len(tiles) > 0
    assert all(
        row_end - row_start <= 256 and col_end - col_start <= 256
        for row_start, row_end, col_start, col_end in tiles
    )


def test_calculate_tile_grid_overlap():
    """Test that tiles have correct overlap."""
    tiler = Tiler(tile_size=256, overlap=0.3)
    tiles = tiler.calculate_tile_grid(1024, 1024)

    if len(tiles) > 1:
        first_tile = tiles[0]
        second_tile = tiles[1]

        row_start1, row_end1, col_start1, col_end1 = first_tile
        row_start2, row_end2, col_start2, col_end2 = second_tile

        if col_start1 == col_start2:
            overlap_rows = min(row_end1, row_end2) - max(row_start1, row_start2)
            expected_overlap = int(256 * 0.3)
            assert overlap_rows >= expected_overlap - 1


def test_tile_raster(sample_raster, tmp_path):
    """Test tiling a raster."""
    output_dir = tmp_path / "tiles"
    tiler = Tiler(tile_size=256, overlap=0.3)

    output_paths = tiler.tile_raster(sample_raster, output_dir)

    assert len(output_paths) > 0
    assert all(path.exists() for path in output_paths)
    # Tiles should be organized in subfolder named after source
    assert (output_dir / "test_raster").exists()

    with rasterio.open(output_paths[0]) as src:
        assert src.width <= 256
        assert src.height <= 256
        assert src.count == 3


def test_tile_alignment(sample_raster, tmp_path):
    """Test that tiles are properly aligned."""
    output_dir = tmp_path / "tiles"
    tiler = Tiler(tile_size=256, overlap=0.3)

    output_paths = tiler.tile_raster(sample_raster, output_dir)

    if len(output_paths) >= 2:
        with rasterio.open(output_paths[0]) as src1, rasterio.open(output_paths[1]) as src2:
            assert src1.crs == src2.crs
            assert src1.transform.a == src2.transform.a
            assert src1.transform.e == src2.transform.e


def test_tile_organization_by_source(sample_raster, tmp_path):
    """Test that tiles are organized in subfolder by source filename."""
    output_dir = tmp_path / "tiles"
    tiler = Tiler(tile_size=256, overlap=0.3)

    output_paths = tiler.tile_raster(sample_raster, output_dir, organize_by_source=True)

    assert len(output_paths) > 0
    # All tiles should be in subfolder named after source
    source_folder = output_dir / "test_raster"
    assert source_folder.exists()
    assert all(path.parent == source_folder for path in output_paths)


def test_tile_no_organization(sample_raster, tmp_path):
    """Test that tiles can be saved directly without subfolder organization."""
    output_dir = tmp_path / "tiles"
    tiler = Tiler(tile_size=256, overlap=0.3)

    output_paths = tiler.tile_raster(sample_raster, output_dir, organize_by_source=False)

    assert len(output_paths) > 0
    # All tiles should be directly in output_dir
    assert all(path.parent == output_dir for path in output_paths)


def test_tile_naming(sample_raster, tmp_path):
    """Test tile filename generation."""
    output_dir = tmp_path / "tiles"
    tiler = Tiler(tile_size=256, overlap=0.3)

    output_paths = tiler.tile_raster(sample_raster, output_dir, base_filename="test")

    # Tiles should be in subfolder named "test" and have pattern tile_XXXX.tif
    assert all(path.parent.name == "test" for path in output_paths)
    assert all("tile_" in path.name for path in output_paths)
    assert any("tile_0000" in path.name for path in output_paths)


def test_tile_small_raster(tmp_path):
    """Test tiling a raster smaller than tile size."""
    raster_path = tmp_path / "small_raster.tif"

    height, width = 100, 100
    data = np.random.randint(0, 255, (1, height, width), dtype=np.uint8)

    with rasterio.open(
        raster_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs='EPSG:4326',
        transform=rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
    ) as dst:
        dst.write(data)

    output_dir = tmp_path / "tiles"
    tiler = Tiler(tile_size=256, overlap=0.3)

    output_paths = tiler.tile_raster(raster_path, output_dir)

    assert len(output_paths) == 1
    # Small raster should also be organized in subfolder
    assert (output_dir / "small_raster").exists()
    with rasterio.open(output_paths[0]) as src:
        assert src.width == 100
        assert src.height == 100


def test_boundary_tiles_are_full_size(tmp_path):
    """Test that boundary tiles are always full-size (256x256)."""
    # Create raster that doesn't divide evenly by tile size
    raster_path = tmp_path / "test_raster.tif"

    # 1024x1024 with 256 tile size = 4x4 grid, but test with odd size
    height, width = 1000, 1000  # Not divisible by 256
    data = np.random.randint(0, 255, (3, height, width), dtype=np.uint8)

    with rasterio.open(
        raster_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3,
        dtype=data.dtype,
        crs='EPSG:4326',
        transform=rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
    ) as dst:
        dst.write(data)

    output_dir = tmp_path / "tiles"
    tiler = Tiler(tile_size=256, overlap=0.3)

    output_paths = tiler.tile_raster(raster_path, output_dir)

    # Verify all tiles are 256x256
    for tile_path in output_paths:
        with rasterio.open(tile_path) as src:
            assert src.width == 256, f"Tile {tile_path.name} width is {src.width}, expected 256"
            assert src.height == 256, f"Tile {tile_path.name} height is {src.height}, expected 256"
