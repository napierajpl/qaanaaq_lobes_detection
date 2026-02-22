"""
Create a segmentation layer from any raster (independent of dataset).
Output is a separate raster (same grid): segment IDs as hint for CNN boundaries.
Can be run on normal imagery or on imagery with parenthesis. Optional 2 scales.
For large rasters, uses block processing to avoid loading full image.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio
from rasterio.windows import Window

from src.data_processing.boundary_extraction import rasterize_boundaries_to_mask
from src.utils.path_utils import get_project_root, resolve_path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, *args, **kwargs):
        return it

SEGMENTATION_NODATA = -9999


def _segment_raster_band(band: np.ndarray, scale: float, sigma: float = 0.8) -> np.ndarray:
    """Run Felzenszwalb on a single band; returns segment IDs (int)."""
    try:
        from skimage.segmentation import felzenszwalb
    except ImportError:
        raise ImportError(
            "scikit-image is required for create_segmentation_layer. "
            "Install with: pip install scikit-image"
        )

    band_float = np.asarray(band, dtype=np.float64)
    if band_float.ndim == 2:
        pass
    elif band_float.ndim == 3:
        band_float = np.mean(band_float, axis=0)
    else:
        raise ValueError("Band must be 2D or 3D")
    seg = felzenszwalb(band_float, scale=scale, sigma=sigma, min_size=20)
    return seg.astype(np.int32)


def _create_segmentation_in_memory(
    data: np.ndarray,
    scale: float,
    scale2: float | None,
    sigma: float,
) -> np.ndarray:
    """Run segmentation on in-memory array; returns (n_bands, H, W) int32."""
    layers = []
    layers.append(_segment_raster_band(data, scale=scale, sigma=sigma))
    if scale2 is not None and scale2 != scale:
        layers.append(_segment_raster_band(data, scale=scale2, sigma=sigma))
    return np.stack(layers, axis=0).astype(np.int32)


def create_segmentation_layer(
    input_raster_path: Path,
    output_path: Path,
    scale: float = 100.0,
    scale2: float | None = None,
    sigma: float = 0.8,
    block_size: int = 4096,
    boundary_path: Path | None = None,
) -> None:
    """
    Read raster, run OBIA-style segmentation (Felzenszwalb), write segment ID layer(s).
    Output has same grid/CRS as input; 1 or 2 bands (segment IDs).
    If raster larger than block_size, processes in blocks to save memory.
    If boundary_path is set, pixels outside the boundary are written as nodata.
    """
    input_raster_path = Path(input_raster_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    boundary_mask = None
    if boundary_path and Path(boundary_path).exists():
        boundary_mask = rasterize_boundaries_to_mask(boundary_path, input_raster_path)

    with rasterio.open(input_raster_path) as src:
        profile = src.profile.copy()
        H, W = src.height, src.width
        n_bands_out = 2 if (scale2 is not None and scale2 != scale) else 1
        profile.update(count=n_bands_out, dtype=np.int32, nodata=SEGMENTATION_NODATA)

        def apply_boundary_mask(block: np.ndarray, win: Window) -> np.ndarray:
            if boundary_mask is None:
                return block
            r, c = int(win.row_off), int(win.col_off)
            h, w = int(win.height), int(win.width)
            mask_block = boundary_mask[r : r + h, c : c + w]
            out = block.copy()
            out[:, ~mask_block.astype(bool)] = SEGMENTATION_NODATA
            return out

        if H <= block_size and W <= block_size:
            if src.count >= 3:
                data = src.read([1, 2, 3])
            else:
                data = src.read()
            out_stack = _create_segmentation_in_memory(data, scale, scale2, sigma)
            out_stack = apply_boundary_mask(out_stack, Window(0, 0, W, H))
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(out_stack)
            return

        windows = [
            Window(c, r, min(block_size, W - c), min(block_size, H - r))
            for r in range(0, H, block_size)
            for c in range(0, W, block_size)
        ]
        n_windows = len(windows)
        with rasterio.open(output_path, "w", **profile) as dst:
            for i, window in enumerate(tqdm(windows, desc="Segmentation blocks")):
                print(f"  Block {i + 1}/{n_windows}: read -> segment -> write", flush=True)
                if src.count >= 3:
                    data = src.read([1, 2, 3], window=window)
                else:
                    data = src.read(window=window)
                out_block = _create_segmentation_in_memory(data, scale, scale2, sigma)
                out_block = apply_boundary_mask(out_block, window)
                dst.write(out_block, window=window)


def main() -> None:
    project_root = get_project_root(Path(__file__))
    default_input = project_root / "data/raw/raster/imagery/qaanaaq_rgb_0_2m.tif"
    default_output = project_root / "data/processed/raster/imagery_segmentation_layer.tif"
    default_boundary = project_root / "data/raw/vector/research_boundary.shp"

    import argparse
    parser = argparse.ArgumentParser(
        description="Create segmentation layer from raster (separate layer for CNN hints).",
    )
    parser.add_argument("-i", "--input", type=Path, default=default_input)
    parser.add_argument("-o", "--output", type=Path, default=default_output)
    parser.add_argument(
        "--scale",
        type=float,
        default=100.0,
        help="Felzenszwalb scale (higher = larger segments)",
    )
    parser.add_argument(
        "--scale2",
        type=float,
        default=None,
        help="Optional second scale for second band",
    )
    parser.add_argument("--sigma", type=float, default=0.8, help="Felzenszwalb sigma")
    parser.add_argument(
        "--block-size",
        type=int,
        default=4096,
        help="Process in blocks of this size when raster is larger (default 4096)",
    )
    parser.add_argument(
        "-b", "--boundary",
        type=Path,
        default=default_boundary,
        help="Limit segmentation to inside this vector (nodata outside). Default: research_boundary.shp",
    )
    args = parser.parse_args()

    input_path = resolve_path(args.input, project_root)
    output_path = resolve_path(args.output, project_root)
    boundary_path = resolve_path(args.boundary, project_root) if args.boundary else None

    if not input_path.exists():
        raise FileNotFoundError(f"Input raster not found: {input_path}")

    create_segmentation_layer(
        input_path,
        output_path,
        scale=args.scale,
        scale2=args.scale2,
        sigma=args.sigma,
        block_size=args.block_size,
        boundary_path=boundary_path,
    )
    print(f"Wrote segmentation layer to {output_path}")


if __name__ == "__main__":
    main()
