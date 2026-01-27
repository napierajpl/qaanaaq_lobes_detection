"""
Tiling utilities for creating overlapping tiles from raster images.
"""
from pathlib import Path
from typing import Tuple, List

import numpy as np
import rasterio

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, *args, **kwargs):
        return iterable


class Tiler:
    """Creates overlapping tiles from raster images."""

    def __init__(self, tile_size: int = 256, overlap: float = 0.3):
        """
        Initialize tiler.

        Args:
            tile_size: Size of each tile in pixels (square tiles)
            overlap: Overlap ratio (0.0 to 1.0), e.g., 0.3 = 30% overlap
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = int(tile_size * (1 - overlap))

    def calculate_tile_grid(
        self, width: int, height: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Calculate tile grid coordinates.

        Ensures all tiles are full-size (tile_size x tile_size) by adjusting
        overlap for boundary tiles. If a tile would be smaller than tile_size
        at the boundary, the previous tile's overlap is increased to ensure
        the last tile is full-size.

        Args:
            width: Raster width in pixels
            height: Raster height in pixels

        Returns:
            List of (row_start, row_end, col_start, col_end) tuples
        """
        tiles = []

        row = 0
        while row < height:
            col = 0
            while col < width:
                # Calculate tile end positions
                row_end = min(row + self.tile_size, height)
                col_end = min(col + self.tile_size, width)

                # Check if this would be a partial tile at boundary
                # If so, adjust to ensure full-size tile by increasing overlap
                if row_end < row + self.tile_size:
                    # Not enough rows left - adjust to make full-size tile
                    row = max(0, height - self.tile_size)
                    row_end = height

                if col_end < col + self.tile_size:
                    # Not enough cols left - adjust to make full-size tile
                    col = max(0, width - self.tile_size)
                    col_end = width

                tiles.append((row, row_end, col, col_end))

                # Move to next tile
                if col_end >= width:
                    break

                # Calculate next column position
                next_col = col + self.stride

                # If next tile would be partial, adjust to ensure it's full-size
                if next_col + self.tile_size > width and next_col < width:
                    # Adjust to ensure last tile is full-size (increases overlap)
                    col = width - self.tile_size
                else:
                    col = next_col

            if row_end >= height:
                break

            # Calculate next row position
            next_row = row + self.stride

            # If next tile would be partial, adjust to ensure it's full-size
            if next_row + self.tile_size > height and next_row < height:
                # Adjust to ensure last tile is full-size (increases overlap)
                row = height - self.tile_size
            else:
                row = next_row

        return tiles

    def extract_tile(
        self,
        src: rasterio.DatasetReader,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
    ) -> Tuple[np.ndarray, rasterio.Affine]:
        """Extract a tile from raster."""
        window = rasterio.windows.Window.from_slices(
            (row_start, row_end),
            (col_start, col_end),
        )

        # Handle VRTs with mixed dtypes by reading bands individually
        # and converting to a common dtype (float32 for compatibility)
        try:
            data = src.read(window=window)
        except ValueError as e:
            if "more than one 'dtype' found" in str(e):
                # Read each band separately and convert to float32
                data_list = []
                for band_idx in range(1, src.count + 1):
                    band_data = src.read(band_idx, window=window)
                    data_list.append(band_data.astype(np.float32))
                data = np.array(data_list)
            else:
                raise

        transform = rasterio.windows.transform(window, src.transform)

        return data, transform

    def create_tile_filename(self, base_path: Path, tile_idx: int) -> Path:
        """Generate tile filename with zero-padded index."""
        stem = base_path.stem
        suffix = base_path.suffix
        return base_path.parent / f"{stem}_tile_{tile_idx:04d}{suffix}"

    def tile_raster(
        self,
        input_raster_path: Path,
        output_dir: Path,
        base_filename: str = None,
        organize_by_source: bool = True,
    ) -> List[Path]:
        """
        Tile a raster into overlapping tiles.

        Args:
            input_raster_path: Path to input raster
            output_dir: Directory to save tiles
            base_filename: Base name for tiles (uses input name if None)
            organize_by_source: If True, create subfolder named after source file

        Returns:
            List of output tile paths
        """
        input_raster_path = Path(input_raster_path)
        output_dir = Path(output_dir)

        if base_filename is None:
            base_filename = input_raster_path.stem

        if organize_by_source:
            output_dir = output_dir / base_filename

        output_dir.mkdir(parents=True, exist_ok=True)

        src = rasterio.open(input_raster_path)
        tiles = self.calculate_tile_grid(src.width, src.height)

        output_paths = []

        # Use tqdm for progress bar if available
        tile_iter = tqdm(enumerate(tiles), total=len(tiles), desc="Creating tiles", unit="tile")

        for idx, (row_start, row_end, col_start, col_end) in tile_iter:
            data, transform = self.extract_tile(src, row_start, row_end, col_start, col_end)

            tile_filename = output_dir / f"tile_{idx:04d}.tif"

            # Determine output dtype - use float32 if data was converted, otherwise use original
            output_dtype = data.dtype
            if output_dtype == np.float32 and src.dtypes[0] != np.float32:
                # Check if we should use the most common dtype from source
                output_dtype = src.dtypes[0]

            with rasterio.open(
                tile_filename,
                'w',
                driver='GTiff',
                height=data.shape[1],
                width=data.shape[2],
                count=src.count,
                dtype=output_dtype,
                crs=src.crs,
                transform=transform,
                nodata=src.nodata,
                compress='lzw',
            ) as dst:
                # Convert data back to output dtype if needed
                if data.dtype != output_dtype:
                    data = data.astype(output_dtype)
                dst.write(data)

            output_paths.append(tile_filename)

        src.close()
        return output_paths
