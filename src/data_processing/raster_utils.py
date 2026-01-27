"""
Raster processing utilities for geospatial operations.
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform, reproject, Resampling
from scipy.ndimage import distance_transform_edt


class ProximityMapGenerator:
    """Generates proximity maps from binary rasters using distance transforms."""
    
    def __init__(self, max_value: int = 20, max_distance: int = 20):
        """
        Initialize proximity map generator.
        
        Args:
            max_value: Maximum proximity value (for lobe pixels)
            max_distance: Maximum distance to consider (pixels beyond this get value 0)
        """
        self.max_value = max_value
        self.max_distance = max_distance
    
    def load_binary_raster(self, raster_path: Path) -> Tuple[np.ndarray, rasterio.DatasetReader]:
        """Load binary raster and return array with metadata."""
        raster_path = Path(raster_path)
        
        if not raster_path.exists():
            raise FileNotFoundError(f"Raster file not found: {raster_path}")
        
        src = rasterio.open(raster_path)
        raster = src.read(1)
        
        return raster, src
    
    def create_binary_mask(self, raster: np.ndarray, lobe_value: int = 1) -> np.ndarray:
        """Create binary mask from raster (lobes = True, background = False)."""
        return raster == lobe_value
    
    def calculate_distance_transform(self, binary_mask: np.ndarray) -> np.ndarray:
        """Calculate Euclidean distance transform from lobe pixels."""
        return distance_transform_edt(~binary_mask)
    
    def apply_decay_function(self, distance_map: np.ndarray) -> np.ndarray:
        """Apply decay function: value = max(0, max_value - distance)."""
        proximity_map = self.max_value - distance_map
        proximity_map = np.maximum(0, proximity_map)
        proximity_map[distance_map > self.max_distance] = 0
        return proximity_map.astype(np.uint8)
    
    def generate_proximity_map(
        self,
        input_raster_path: Path,
        output_raster_path: Path,
        lobe_value: int = 1,
    ) -> None:
        """
        Generate proximity map from binary raster.
        
        Args:
            input_raster_path: Path to input binary raster (lobes = lobe_value, background = 0)
            output_raster_path: Path to output proximity map
            lobe_value: Value representing lobes in input raster
        """
        raster, src = self.load_binary_raster(input_raster_path)
        binary_mask = self.create_binary_mask(raster, lobe_value)
        distance_map = self.calculate_distance_transform(binary_mask)
        proximity_map = self.apply_decay_function(distance_map)
        
        output_raster_path.parent.mkdir(parents=True, exist_ok=True)
        
        with rasterio.open(
            output_raster_path,
            'w',
            driver='GTiff',
            height=src.height,
            width=src.width,
            count=1,
            dtype=proximity_map.dtype,
            crs=src.crs,
            transform=src.transform,
            nodata=0,
            compress='lzw',
        ) as dst:
            dst.write(proximity_map, 1)
        
        src.close()


def generate_proximity_map(
    input_raster_path: Path,
    output_raster_path: Path,
    max_value: int = 20,
    max_distance: int = 20,
    lobe_value: int = 1,
) -> None:
    """
    Convenience function for generating proximity maps.
    
    Args:
        input_raster_path: Path to input binary raster
        output_raster_path: Path to output proximity map
        max_value: Maximum proximity value (for lobe pixels)
        max_distance: Maximum distance to consider
        lobe_value: Value representing lobes in input raster
    """
    generator = ProximityMapGenerator(max_value=max_value, max_distance=max_distance)
    generator.generate_proximity_map(input_raster_path, output_raster_path, lobe_value)


class RasterCropper:
    """Crops raster images from specified coordinates and dimensions."""
    
    def load_raster(self, raster_path: Path) -> rasterio.DatasetReader:
        """Load raster file and return dataset reader."""
        raster_path = Path(raster_path)
        
        if not raster_path.exists():
            raise FileNotFoundError(f"Raster file not found: {raster_path}")
        
        return rasterio.open(raster_path)
    
    def world_to_pixel(self, x: float, y: float, transform: rasterio.Affine) -> Tuple[int, int]:
        """Convert world coordinates to pixel coordinates."""
        row, col = rasterio.transform.rowcol(transform, x, y)
        return int(row), int(col)
    
    def create_crop_transform(
        self,
        top_left_x: float,
        top_left_y: float,
        src_transform: rasterio.Affine,
    ) -> rasterio.Affine:
        """Create transform for cropped raster with exact top-left coordinates."""
        return rasterio.Affine(
            src_transform.a,
            src_transform.b,
            top_left_x,
            src_transform.d,
            src_transform.e,
            top_left_y,
        )
    
    def calculate_crop_window(
        self,
        top_left_x: float,
        top_left_y: float,
        width_pixels: int,
        height_pixels: int,
        src: rasterio.DatasetReader,
    ) -> Tuple[int, int, int, int]:
        """
        Calculate crop window in pixel coordinates.
        
        Args:
            top_left_x: Top-left X coordinate (world coordinates)
            top_left_y: Top-left Y coordinate (world coordinates)
            width_pixels: Width of crop in pixels
            height_pixels: Height of crop in pixels
            src: Source raster dataset
        
        Returns:
            Tuple of (row_start, row_stop, col_start, col_stop)
        """
        row_start, col_start = self.world_to_pixel(top_left_x, top_left_y, src.transform)
        row_stop = row_start + height_pixels
        col_stop = col_start + width_pixels
        
        if row_start < 0 or col_start < 0:
            raise ValueError(
                f"Crop coordinates ({top_left_x}, {top_left_y}) are outside raster bounds. "
                f"Raster bounds: {src.bounds}, Calculated pixel: row={row_start}, col={col_start}"
            )
        
        if row_stop > src.height or col_stop > src.width:
            raise ValueError(
                f"Crop dimensions ({width_pixels}x{height_pixels}) exceed raster bounds. "
                f"Raster size: {src.width}x{src.height}, "
                f"Crop window: rows {row_start}-{row_stop}, cols {col_start}-{col_stop}"
            )
        
        return row_start, row_stop, col_start, col_stop
    
    def crop_raster_window(
        self,
        src: rasterio.DatasetReader,
        row_start: int,
        row_stop: int,
        col_start: int,
        col_stop: int,
    ) -> Tuple[np.ndarray, rasterio.Affine]:
        """Crop raster window and return data with updated transform."""
        window = rasterio.windows.Window.from_slices(
            (row_start, row_stop),
            (col_start, col_stop),
        )
        
        data = src.read(window=window)
        transform = rasterio.windows.transform(window, src.transform)
        
        return data, transform
    
    def generate_output_filename(
        self, input_path: Path, width: int, height: int, output_path: Path = None
    ) -> Path:
        """Generate output filename with crop dimensions suffix."""
        if output_path:
            return output_path
        
        input_path = Path(input_path)
        stem = input_path.stem
        suffix = f"_cropped{width}x{height}"
        return input_path.parent / f"{stem}{suffix}{input_path.suffix}"
    
    def convert_geographic_to_projected(
        self, lon: float, lat: float, target_crs: rasterio.crs.CRS
    ) -> Tuple[float, float]:
        """Convert geographic coordinates (lon, lat) to projected coordinates."""
        src_crs = rasterio.crs.CRS.from_epsg(4326)
        x, y = warp_transform(src_crs, target_crs, [lon], [lat])
        return x[0], y[0]
    
    def crop_raster(
        self,
        input_raster_path: Path,
        top_left_x: float,
        top_left_y: float,
        width_pixels: int,
        height_pixels: int,
        output_raster_path: Path = None,
        use_geographic: bool = False,
    ) -> Path:
        """
        Crop raster from specified coordinates and dimensions.
        
        Args:
            input_raster_path: Path to input raster
            top_left_x: Top-left X coordinate (world coordinates or longitude if use_geographic=True)
            top_left_y: Top-left Y coordinate (world coordinates or latitude if use_geographic=True)
            width_pixels: Width of crop in pixels
            height_pixels: Height of crop in pixels
            output_raster_path: Optional output path (auto-generated if None)
            use_geographic: If True, interpret coordinates as lon/lat (EPSG:4326)
        
        Returns:
            Path to output cropped raster
        """
        src = self.load_raster(input_raster_path)
        
        if use_geographic:
            top_left_x, top_left_y = self.convert_geographic_to_projected(
                top_left_x, top_left_y, src.crs
            )
        
        row_start, row_stop, col_start, col_stop = self.calculate_crop_window(
            top_left_x, top_left_y, width_pixels, height_pixels, src
        )
        
        data, _ = self.crop_raster_window(src, row_start, row_stop, col_start, col_stop)
        
        if data.size > 0 and data.max() == 255 and data.min() == 255:
            import warnings
            warnings.warn(
                f"Warning: Cropped data appears to be all white (255). "
                f"Top-left pixel: {data[:, 0, 0] if data.shape[0] > 0 else 'N/A'}"
            )
        
        crop_transform = self.create_crop_transform(
            top_left_x,
            top_left_y,
            src.transform,
        )
        
        output_path = self.generate_output_filename(
            input_raster_path, width_pixels, height_pixels, output_raster_path
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height_pixels,
            width=width_pixels,
            count=src.count,
            dtype=data.dtype,
            crs=src.crs,
            transform=crop_transform,
            nodata=src.nodata,
            compress='lzw',
        ) as dst:
            dst.write(data, indexes=list(range(1, src.count + 1)))
        
        src.close()
        return output_path


def crop_raster(
    input_raster_path: Path,
    top_left_x: float,
    top_left_y: float,
    width_pixels: int,
    height_pixels: int,
    output_raster_path: Path = None,
    use_geographic: bool = False,
) -> Path:
    """
    Convenience function for cropping rasters.
    
    Args:
        input_raster_path: Path to input raster
        top_left_x: Top-left X coordinate (world coordinates or longitude if use_geographic=True)
        top_left_y: Top-left Y coordinate (world coordinates or latitude if use_geographic=True)
        width_pixels: Width of crop in pixels
        height_pixels: Height of crop in pixels
        output_raster_path: Optional output path (auto-generated if None)
        use_geographic: If True, interpret coordinates as lon/lat (EPSG:4326)
    
    Returns:
        Path to output cropped raster
    """
    cropper = RasterCropper()
    return cropper.crop_raster(
        input_raster_path,
        top_left_x,
        top_left_y,
        width_pixels,
        height_pixels,
        output_raster_path,
        use_geographic,
    )


class VirtualRasterStacker:
    """Creates virtual raster (VRT) files that reference multiple rasters without copying data."""
    
    def validate_compatibility(self, raster_paths: list[Path]) -> None:
        """Validate that all rasters have compatible dimensions and CRS."""
        if not raster_paths:
            raise ValueError("No raster paths provided")
        
        first_src = rasterio.open(raster_paths[0])
        first_shape = (first_src.height, first_src.width)
        first_crs = first_src.crs
        first_transform = first_src.transform
        
        for i, path in enumerate(raster_paths[1:], 1):
            src = rasterio.open(path)
            if (src.height, src.width) != first_shape:
                src.close()
                first_src.close()
                raise ValueError(
                    f"Raster {i} ({path}) has incompatible shape: "
                    f"{(src.height, src.width)} vs {first_shape}"
                )
            if src.crs != first_crs:
                src.close()
                first_src.close()
                raise ValueError(
                    f"Raster {i} ({path}) has incompatible CRS: {src.crs} vs {first_crs}"
                )
            if src.transform != first_transform:
                src.close()
                first_src.close()
                raise ValueError(
                    f"Raster {i} ({path}) has incompatible transform. Consider resampling first."
                )
            src.close()
        
        first_src.close()
    
    def create_vrt_stack(
        self,
        raster_paths: list[Path],
        output_path: Path,
    ) -> Path:
        """
        Create a VRT file that stacks multiple rasters as bands.
        
        Args:
            raster_paths: List of input raster file paths
            output_path: Output VRT file path
        
        Returns:
            Path to output VRT file
        """
        if not raster_paths:
            raise ValueError("No raster paths provided")
        
        self.validate_compatibility(raster_paths)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        first_src = rasterio.open(raster_paths[0])
        total_bands = sum(rasterio.open(path).count for path in raster_paths)
        
        vrt_content = self._generate_vrt_xml(raster_paths, first_src, total_bands)
        first_src.close()
        
        output_path.write_text(vrt_content, encoding='utf-8')
        
        return output_path
    
    def _generate_vrt_xml(
        self,
        raster_paths: list[Path],
        reference_src: rasterio.DatasetReader,
        total_bands: int,
    ) -> str:
        """Generate VRT XML content for stacked rasters."""
        transform = reference_src.transform
        
        xml_parts = [
            '<VRTDataset rasterXSize="{}" rasterYSize="{}">'.format(
                reference_src.width, reference_src.height
            ),
            f'  <SRS>{reference_src.crs.to_string() if reference_src.crs else ""}</SRS>',
            '  <GeoTransform>{}, {}, {}, {}, {}, {}</GeoTransform>'.format(
                transform.c,
                transform.a,
                transform.b,
                transform.f,
                transform.d,
                transform.e,
            ),
        ]
        
        band_idx = 1
        for raster_path in raster_paths:
            src = rasterio.open(raster_path)
            abs_path = Path(raster_path).absolute().as_posix()
            
            for band_num in range(1, src.count + 1):
                xml_parts.append(
                    f'  <VRTRasterBand dataType="{self._get_gdal_datatype(src.dtypes[band_num-1])}" '
                    f'band="{band_idx}">'
                )
                xml_parts.append('    <SimpleSource>')
                xml_parts.append(f'      <SourceFilename relativeToVRT="0">{abs_path}</SourceFilename>')
                xml_parts.append(f'      <SourceBand>{band_num}</SourceBand>')
                xml_parts.append(f'      <SourceProperties RasterXSize="{src.width}" '
                               f'RasterYSize="{src.height}" '
                               f'DataType="{self._get_gdal_datatype(src.dtypes[band_num-1])}" '
                               f'BlockXSize="{src.block_shapes[0][1]}" '
                               f'BlockYSize="{src.block_shapes[0][0]}"/>')
                xml_parts.append(f'      <SrcRect xOff="0" yOff="0" xSize="{src.width}" ySize="{src.height}"/>')
                xml_parts.append(f'      <DstRect xOff="0" yOff="0" xSize="{src.width}" ySize="{src.height}"/>')
                xml_parts.append('    </SimpleSource>')
                xml_parts.append('  </VRTRasterBand>')
                band_idx += 1
            
            src.close()
        
        xml_parts.append('</VRTDataset>')
        
        return '\n'.join(xml_parts)
    
    def _get_gdal_datatype(self, numpy_dtype) -> str:
        """Convert numpy dtype to GDAL data type string."""
        dtype_map = {
            'uint8': 'Byte',
            'uint16': 'UInt16',
            'int16': 'Int16',
            'uint32': 'UInt32',
            'int32': 'Int32',
            'float32': 'Float32',
            'float64': 'Float64',
        }
        return dtype_map.get(str(numpy_dtype), 'Byte')


def create_vrt_stack(raster_paths: list[Path], output_path: Path) -> Path:
    """
    Convenience function for creating VRT stacks.
    
    Args:
        raster_paths: List of input raster file paths
        output_path: Output VRT file path
    
    Returns:
        Path to output VRT file
    """
    stacker = VirtualRasterStacker()
    return stacker.create_vrt_stack(raster_paths, output_path)


def resample_raster_to_match(
    input_raster_path: Path,
    reference_raster_path: Path,
    output_raster_path: Path,
    resampling_method: Resampling = Resampling.bilinear,
) -> Path:
    """
    Resample a raster to match another raster's transform, CRS, and dimensions.
    
    Args:
        input_raster_path: Path to input raster to resample
        reference_raster_path: Path to reference raster to match
        output_raster_path: Path to output resampled raster
        resampling_method: Resampling algorithm (default: bilinear)
    
    Returns:
        Path to output raster
    """
    input_raster_path = Path(input_raster_path)
    reference_raster_path = Path(reference_raster_path)
    output_raster_path = Path(output_raster_path)
    
    output_raster_path.parent.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(reference_raster_path) as ref_src:
        ref_transform = ref_src.transform
        ref_crs = ref_src.crs
        ref_width = ref_src.width
        ref_height = ref_src.height
    
    with rasterio.open(input_raster_path) as src:
        # Read all bands
        data = src.read()
        num_bands = src.count
        
        # Create output array
        output_data = np.zeros((num_bands, ref_height, ref_width), dtype=data.dtype)
        
        # Reproject each band
        for band_idx in range(num_bands):
            reproject(
                source=data[band_idx],
                destination=output_data[band_idx],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=resampling_method,
            )
        
        # Write output
        with rasterio.open(
            output_raster_path,
            'w',
            driver='GTiff',
            height=ref_height,
            width=ref_width,
            count=num_bands,
            dtype=data.dtype,
            crs=ref_crs,
            transform=ref_transform,
            nodata=src.nodata,
            compress='lzw',
        ) as dst:
            dst.write(output_data)
    
    return output_raster_path
