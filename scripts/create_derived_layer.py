#!/usr/bin/env python3
"""Generic CLI for creating a derived layer raster from the transform registry.

Usage:
    poetry run python scripts/create_derived_layer.py \
        --transform slope_stripes \
        --input rgb=data/raw/raster/imagery/qaanaaq_rgb_0_2m.tif \
        --input dem=data/processed/raster/dem_from_arcticDEM_resampled.tif \
        --output data/processed/raster/slope_stripes_channel.tif \
        --param method=gabor --param frequency=0.15
"""

import argparse
from pathlib import Path

from src.preprocessing.derived_transforms import TRANSFORM_REGISTRY, create_derived_layer_raster
from src.utils.config_utils import parse_param_value


def main():
    parser = argparse.ArgumentParser(description="Create a derived layer raster")
    parser.add_argument("--transform", required=True, choices=list(TRANSFORM_REGISTRY.keys()))
    parser.add_argument("--input", action="append", required=True, metavar="NAME=PATH",
                        help="Input raster: name=path (repeat for each input)")
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                        help="Transform parameter (repeat for each)")
    parser.add_argument("--block-size", type=int, default=2048)
    args = parser.parse_args()

    input_paths = {}
    for kv in args.input:
        name, _, path = kv.partition("=")
        if not path:
            parser.error(f"--input must be NAME=PATH, got: {kv}")
        input_paths[name.strip()] = Path(path.strip())

    params = {}
    for kv in args.param:
        key, _, val = kv.partition("=")
        if not val:
            parser.error(f"--param must be KEY=VALUE, got: {kv}")
        params[key.strip()] = parse_param_value(val.strip())

    transform = TRANSFORM_REGISTRY[args.transform]
    print(f"Creating derived layer: {args.transform}")
    print(f"  Inputs: {input_paths}")
    print(f"  Output: {args.output}")
    print(f"  Params: {params}")

    create_derived_layer_raster(
        transform=transform,
        input_paths=input_paths,
        output_path=args.output,
        params=params,
        block_size=args.block_size,
    )
    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()
