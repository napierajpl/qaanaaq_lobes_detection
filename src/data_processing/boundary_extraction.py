"""
Extract valid-data boundary polygons from a raster and write as vector.
Used to constrain placement (e.g. parenthesis) and for other masking/QC.
"""
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import shape


def build_valid_mask(
    rgb: np.ndarray,
    white_threshold: int = 250,
    nodata_values: Optional[list] = None,
) -> np.ndarray:
    """Build binary mask: 1 = valid (not white, not nodata), 0 = invalid."""
    if rgb.shape[0] < 3:
        return np.zeros((rgb.shape[1], rgb.shape[2]), dtype=np.uint8)
    r, g, b = rgb[0], rgb[1], rgb[2]
    not_white = ~((r >= white_threshold) & (g >= white_threshold) & (b >= white_threshold))
    valid = not_white.astype(np.uint8)
    if nodata_values is not None and len(nodata_values) >= 3:
        for band_idx, nd in enumerate(nodata_values[:3]):
            if nd is not None:
                valid = valid & (rgb[band_idx] != nd)
    return valid


def extract_boundaries_from_raster(
    raster_path: Path,
    white_threshold: int = 250,
    use_nodata: bool = False,
) -> gpd.GeoDataFrame:
    """
    Extract boundary polygons for valid-data regions (non-white, optional non-nodata).
    Returns GeoDataFrame with geometry in raster CRS.
    """
    path = Path(raster_path)
    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")

    with rasterio.open(path) as src:
        if src.count < 3:
            raise ValueError(f"Raster must have at least 3 bands (RGB), got {src.count}")
        rgb = src.read([1, 2, 3])
        transform = src.transform
        crs = src.crs
        nodata_values = [src.nodata] * 3 if use_nodata and src.nodata is not None else None

    mask = build_valid_mask(rgb, white_threshold=white_threshold, nodata_values=nodata_values)

    geoms = []
    for geom_dict, value in features.shapes(mask, transform=transform):
        if value != 1:
            continue
        try:
            geoms.append(shape(geom_dict))
        except Exception:
            continue

    if not geoms:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    return gdf


def write_boundaries_vector(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """Write GeoDataFrame to GeoJSON or shapefile."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GeoJSON" if output_path.suffix.lower() == ".geojson" else "ESRI Shapefile")


def rasterize_boundaries_to_mask(
    vector_path: Path,
    reference_raster_path: Path,
    all_touched: bool = True,
) -> np.ndarray:
    """
    Rasterize boundary vector to binary mask (same shape as reference raster).
    1 = inside boundary, 0 = outside. Use for point-in-polygon sampling.
    """
    from src.data_processing.vector_utils import Rasterizer

    rasterizer = Rasterizer(burn_value=1, nodata=0, all_touched=all_touched)
    gdf = rasterizer.load_vector(vector_path)
    with rasterio.open(reference_raster_path) as ref:
        gdf = rasterizer.reproject_if_needed(gdf, ref.crs)
    _, _, (width, height), _, transform = rasterizer.prepare_raster_specs(gdf, reference_raster_path)
    arr = rasterizer.create_raster_array(gdf, height, width, transform)
    return (arr == 1).astype(np.uint8)
