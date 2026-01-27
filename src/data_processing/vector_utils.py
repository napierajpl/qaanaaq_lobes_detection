from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import from_bounds


class Rasterizer:
    """Rasterizes vector geometries to raster format."""

    def __init__(
        self,
        burn_value: int = 1,
        nodata: int = 0,
        all_touched: bool = False,
    ):
        """
        Initialize rasterizer with default settings.

        Args:
            burn_value: Value to burn for vector features
            nodata: NoData value for output raster
            all_touched: If True, all pixels touched by geometry will be burned
        """
        self.burn_value = burn_value
        self.nodata = nodata
        self.all_touched = all_touched

    def load_vector(self, vector_path: Path) -> gpd.GeoDataFrame:
        """Load and validate vector file."""
        vector_path = Path(vector_path)

        if not vector_path.exists():
            raise FileNotFoundError(f"Vector file not found: {vector_path}")

        gdf = gpd.read_file(vector_path)

        if gdf.empty:
            raise ValueError(f"Vector file is empty: {vector_path}")

        return gdf

    def get_reference_info(
        self, reference_raster_path: Path
    ) -> Tuple[Tuple[float, float, float, float], Tuple[int, int], rasterio.crs.CRS, rasterio.Affine]:
        """Extract extent, shape, CRS, and transform from a reference raster."""
        with rasterio.open(reference_raster_path) as src:
            return src.bounds, src.shape, src.crs, src.transform

    def reproject_if_needed(self, gdf: gpd.GeoDataFrame, target_crs: rasterio.crs.CRS) -> gpd.GeoDataFrame:
        """Reproject GeoDataFrame if CRS doesn't match target."""
        if gdf.crs != target_crs:
            return gdf.to_crs(target_crs)
        return gdf

    def calculate_raster_specs_from_bounds(
        self, bounds: Tuple[float, float, float, float], crs: rasterio.crs.CRS, resolution: float
    ) -> Tuple[int, int, rasterio.Affine]:
        """Calculate raster dimensions and transform from bounds and resolution."""
        width = int((bounds[2] - bounds[0]) / resolution)
        height = int((bounds[3] - bounds[1]) / resolution)
        transform = from_bounds(*bounds, width, height)
        return width, height, transform

    def prepare_raster_specs(
        self,
        gdf: gpd.GeoDataFrame,
        reference_raster_path: Optional[Path] = None,
        resolution: Optional[float] = None,
    ) -> Tuple[gpd.GeoDataFrame, Tuple[float, float, float, float], Tuple[int, int], rasterio.crs.CRS, rasterio.Affine]:
        """Determine raster specifications from reference or vector bounds."""
        if reference_raster_path and Path(reference_raster_path).exists():
            bounds, shape, crs, transform = self.get_reference_info(reference_raster_path)
            width, height = shape[1], shape[0]
            gdf = self.reproject_if_needed(gdf, crs)
        else:
            bounds = gdf.total_bounds
            crs = gdf.crs

            if crs is None:
                raise ValueError("Vector file has no CRS defined")

            if resolution is None:
                resolution = 1.0

            width, height, transform = self.calculate_raster_specs_from_bounds(bounds, crs, resolution)

        return gdf, bounds, (width, height), crs, transform

    def create_raster_array(
        self, gdf: gpd.GeoDataFrame, height: int, width: int, transform: rasterio.Affine
    ) -> np.ndarray:
        """Create raster array from vector geometries."""
        shapes = ((geom, self.burn_value) for geom in gdf.geometry)

        return features.rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=self.nodata,
            all_touched=self.all_touched,
            dtype=np.uint8,
        )

    def write_raster(
        self,
        output_path: Path,
        raster: np.ndarray,
        width: int,
        height: int,
        crs: rasterio.crs.CRS,
        transform: rasterio.Affine,
    ) -> None:
        """Write raster array to GeoTIFF file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=raster.dtype,
            crs=crs,
            transform=transform,
            nodata=self.nodata,
            compress='lzw',
        ) as dst:
            dst.write(raster, 1)

    def rasterize(
        self,
        vector_path: Path,
        output_path: Path,
        reference_raster_path: Optional[Path] = None,
        resolution: Optional[float] = None,
    ) -> None:
        """
        Rasterize a vector layer to a GeoTIFF file.

        Args:
            vector_path: Path to input vector file (shapefile, GeoJSON, etc.)
            output_path: Path to output raster file
            reference_raster_path: Optional reference raster to match extent/resolution
            resolution: Pixel resolution in CRS units (used if no reference raster)
        """
        gdf = self.load_vector(vector_path)
        gdf, bounds, (width, height), crs, transform = self.prepare_raster_specs(gdf, reference_raster_path, resolution)
        raster = self.create_raster_array(gdf, height, width, transform)
        self.write_raster(output_path, raster, width, height, crs, transform)


def rasterize_vector(
    vector_path: Path,
    output_path: Path,
    reference_raster_path: Optional[Path] = None,
    burn_value: int = 1,
    nodata: int = 0,
    all_touched: bool = False,
    resolution: Optional[float] = None,
) -> None:
    """
    Convenience function for rasterizing vectors.

    Args:
        vector_path: Path to input vector file (shapefile, GeoJSON, etc.)
        output_path: Path to output raster file
        reference_raster_path: Optional reference raster to match extent/resolution
        burn_value: Value to burn for vector features (default: 1)
        nodata: NoData value for output raster (default: 0)
        all_touched: If True, all pixels touched by geometry will be burned
        resolution: Pixel resolution in CRS units (used if no reference raster)
    """
    rasterizer = Rasterizer(burn_value=burn_value, nodata=nodata, all_touched=all_touched)
    rasterizer.rasterize(vector_path, output_path, reference_raster_path, resolution)
