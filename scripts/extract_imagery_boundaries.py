"""
Extract valid-data boundary polygons from the reference RGB raster and write vector.
Run once; reuse for synthetic parenthesis placement and other masking/QC.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_processing.boundary_extraction import (
    extract_boundaries_from_raster,
    write_boundaries_vector,
)
from src.utils.path_utils import get_project_root, resolve_path


def main() -> None:
    project_root = get_project_root(Path(__file__))
    default_raster = project_root / "data/raw/raster/imagery/qaanaaq_rgb_0_2m.tif"
    default_output = project_root / "data/processed/vector/imagery_valid_boundaries.geojson"

    import argparse
    parser = argparse.ArgumentParser(
        description="Extract valid-data boundary polygons from raster (non-white regions).",
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=default_raster,
        help=f"Input RGB raster (default: {default_raster})",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=default_output,
        help=f"Output vector GeoJSON (default: {default_output})",
    )
    parser.add_argument(
        "--white-threshold",
        type=int,
        default=250,
        help="Pixel is white if all RGB >= this (default: 250)",
    )
    parser.add_argument(
        "--use-nodata",
        action="store_true",
        help="Exclude nodata pixels from valid regions",
    )
    args = parser.parse_args()

    input_path = resolve_path(args.input, project_root)
    output_path = resolve_path(args.output, project_root)

    if not input_path.exists():
        raise FileNotFoundError(f"Input raster not found: {input_path}")

    gdf = extract_boundaries_from_raster(
        input_path,
        white_threshold=args.white_threshold,
        use_nodata=args.use_nodata,
    )
    write_boundaries_vector(gdf, output_path)
    print(f"Wrote {len(gdf)} polygon(s) to {output_path}")


if __name__ == "__main__":
    main()
