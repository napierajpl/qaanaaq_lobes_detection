"""Derived-layer transforms: config-driven raster transformations with chunked I/O.

A *derived layer* is a raster channel computed from one or more existing rasters
(e.g. slope_stripes = f(RGB, DEM)).  Each transform is a subclass of
``DerivedLayerTransform`` that implements ``compute_block``.

To add a new derived layer:
1. Subclass ``DerivedLayerTransform`` and implement ``compute_block``.
2. Register it: ``TRANSFORM_REGISTRY["my_layer"] = MyTransform()``.
3. Add config in ``data_pipeline_paths.yaml`` under ``derived_layers``.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import numpy as np
import rasterio
from rasterio.windows import Window

from src.preprocessing.texture_hints import (
    compute_gabor_slope_stripes_channel,
    compute_slope_stripes_channel,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, *args, **kwargs):  # type: ignore[misc]
        return it


class DerivedLayerTransform(ABC):
    name: str

    @property
    @abstractmethod
    def input_names(self) -> List[str]:
        """Ordered list of required input names (e.g. ["rgb", "dem"])."""

    @property
    @abstractmethod
    def input_band_counts(self) -> Dict[str, int]:
        """Map input name → expected number of bands."""

    @abstractmethod
    def compute_block(
        self, inputs: Dict[str, np.ndarray], params: dict
    ) -> np.ndarray:
        """Compute one block. Returns (H, W) float32 array in [0, 1]."""


class SlopeStripesTransform(DerivedLayerTransform):
    name = "slope_stripes"

    @property
    def input_names(self) -> List[str]:
        return ["rgb", "dem"]

    @property
    def input_band_counts(self) -> Dict[str, int]:
        return {"rgb": 3, "dem": 1}

    def compute_block(
        self, inputs: Dict[str, np.ndarray], params: dict
    ) -> np.ndarray:
        rgb = inputs["rgb"]
        dem = inputs["dem"]
        method = params.get("method", "gabor")
        if method == "structure_tensor":
            return compute_slope_stripes_channel(
                rgb, dem,
                sigma_smooth=params.get("sigma_smooth", 1.5),
                sigma_structure=params.get("sigma_structure", 2.0),
                alignment_power=params.get("alignment_power", 1.0),
            )
        if method == "gabor":
            return compute_gabor_slope_stripes_channel(
                rgb, dem,
                frequency=params.get("frequency", 0.15),
                sigma=params.get("sigma", 5.0),
                n_orientations=params.get("n_orientations", 16),
                alignment_power=params.get("alignment_power", 1.0),
            )
        raise ValueError(f"Unknown slope_stripes method: {method}")


TRANSFORM_REGISTRY: Dict[str, DerivedLayerTransform] = {
    "slope_stripes": SlopeStripesTransform(),
}


def create_derived_layer_raster(
    transform: DerivedLayerTransform,
    input_paths: Dict[str, Path],
    output_path: Path,
    params: dict,
    block_size: int = 2048,
) -> None:
    """Chunked raster creation for any derived layer transform.

    Opens all input rasters, iterates over blocks, calls
    ``transform.compute_block``, and writes a single-band float32 output.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sources = {
        name: rasterio.open(Path(input_paths[name]))
        for name in transform.input_names
    }

    try:
        ref_name = transform.input_names[0]
        ref = sources[ref_name]
        H, W = ref.height, ref.width
        for name, src in sources.items():
            if (src.height, src.width) != (H, W):
                raise ValueError(
                    f"Input size mismatch: {ref_name} is {H}x{W} but "
                    f"{name} is {src.height}x{src.width}. "
                    "Resample inputs to the same grid first."
                )

        profile = ref.profile.copy()
        profile.update(count=1, dtype=np.float32, nodata=None)

        if H <= block_size and W <= block_size:
            inputs = _read_block(sources, transform, window=None)
            out_band = transform.compute_block(inputs, params)
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(out_band.astype(np.float32), 1)
            return

        windows = [
            Window(c, r, min(block_size, W - c), min(block_size, H - r))
            for r in range(0, H, block_size)
            for c in range(0, W, block_size)
        ]
        with rasterio.open(output_path, "w", **profile) as dst:
            for window in tqdm(windows, desc=f"{transform.name} blocks"):
                inputs = _read_block(sources, transform, window)
                out_block = transform.compute_block(inputs, params)
                dst.write(out_block.astype(np.float32), 1, window=window)
    finally:
        for src in sources.values():
            src.close()


def _read_block(
    sources: Dict[str, rasterio.DatasetReader],
    transform: DerivedLayerTransform,
    window,
) -> Dict[str, np.ndarray]:
    inputs: Dict[str, np.ndarray] = {}
    for name in transform.input_names:
        n_bands = transform.input_band_counts[name]
        src = sources[name]
        if n_bands == 1:
            inputs[name] = src.read(1, window=window)
        else:
            bands = list(range(1, n_bands + 1))
            inputs[name] = src.read(bands, window=window)
    return inputs
