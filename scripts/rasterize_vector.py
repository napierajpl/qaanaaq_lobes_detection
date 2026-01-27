#!/usr/bin/env python3
"""Script to rasterize vector layer to GeoTIFF."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.vector_utils import rasterize_vector
from src.utils.cli_utils import BaseCLIParser
from src.utils.path_utils import get_project_root, resolve_path


def main():
    """Rasterize the vector layer."""
    project_root = get_project_root(__file__)
    
    parser = BaseCLIParser(
        description="Rasterize a vector layer to GeoTIFF format",
        project_root=project_root,
    )
    
    default_vector = project_root / "data" / "raw" / "vector" / "loby liniowe.shp"
    default_output = project_root / "data" / "processed" / "raster" / "rasterized_lobes_raw_by_code.tif"
    default_reference = project_root / "data" / "processed" / "raster" / "rasterized_lobes_raw.tif"
    
    parser.add_input_output_args(
        default_input=default_vector,
        default_output=default_output,
    )
    parser.add_reference_raster_arg(default_reference=default_reference)
    
    parser.set_epilog("""
Examples:
  # Use default paths
  python scripts/rasterize_vector.py
  
  # Specify custom input and output
  python scripts/rasterize_vector.py -i data/raw/vector/my_layer.shp -o output.tif
  
  # With custom reference raster
  python scripts/rasterize_vector.py -i input.shp -o output.tif -r reference.tif
    """)
    
    args = parser.parse_args()
    
    vector_path = resolve_path(args.input, project_root)
    output_path = resolve_path(args.output, project_root)
    
    if args.no_reference:
        reference_raster_path = None
    else:
        reference_raster_path = resolve_path(args.reference, project_root)
        if not reference_raster_path.exists():
            reference_raster_path = None
    
    print(f"Rasterizing vector: {vector_path}")
    print(f"Output: {output_path}")
    
    if reference_raster_path:
        print(f"Using reference raster for extent/resolution: {reference_raster_path}")
        rasterize_vector(
            vector_path=vector_path,
            output_path=output_path,
            reference_raster_path=reference_raster_path,
        )
    else:
        print("No reference raster found, using vector bounds")
        rasterize_vector(
            vector_path=vector_path,
            output_path=output_path,
        )
    
    print(f"Rasterization complete: {output_path}")


if __name__ == "__main__":
    main()

