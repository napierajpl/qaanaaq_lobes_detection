"""
Create the full segmentation raster from the synthetic RGB-with-shapes layer
(data/processed/raster/synthetic_parenthesis/synthetic_rgb_with_shapes.tif).
Output: data/processed/raster/synthetic_parenthesis/segmentation_layer.tif (same grid as input).

Tiling of this raster to 256/512 is done by generate_synthetic_parenthesis_from_raster.py
or by scripts/tile_synthetic_segmentation.py.

Requires: synthetic parenthesis full layers in data/processed/raster/synthetic_parenthesis/
(generate_synthetic_parenthesis_from_raster.py writes them there).
Requires: pip install scikit-image (for create_segmentation_layer).
"""
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.path_utils import get_project_root, resolve_path

OLD_RASTER_DIR = "data/processed/tiles/raster"
SYNTHETIC_LAYERS = ["synthetic_rgb_with_shapes.tif", "synthetic_features_5band.tif", "synthetic_target.tif"]


def _move_layers_from_tiles_raster_if_needed(project_root: Path, new_dir: Path) -> None:
    old_dir = project_root / OLD_RASTER_DIR
    if not old_dir.exists():
        return
    if new_dir.exists() and (new_dir / SYNTHETIC_LAYERS[0]).exists():
        return
    new_dir.mkdir(parents=True, exist_ok=True)
    for name in SYNTHETIC_LAYERS:
        src = old_dir / name
        dst = new_dir / name
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            print(f"Moved {src.relative_to(project_root)} -> {dst.relative_to(project_root)}")


def main() -> None:
    project_root = get_project_root(Path(__file__))
    default_raster = project_root / "data/processed/raster/synthetic_parenthesis/synthetic_rgb_with_shapes.tif"
    default_boundary = project_root / "data/raw/vector/research_boundary.shp"
    default_seg_raster = project_root / "data/processed/raster/synthetic_parenthesis/segmentation_layer.tif"

    import argparse
    parser = argparse.ArgumentParser(
        description="Create full segmentation raster for synthetic parenthesis (no tiling).",
    )
    parser.add_argument("-i", "--raster", type=Path, default=default_raster)
    parser.add_argument("-b", "--boundary", type=Path, default=default_boundary)
    parser.add_argument(
        "-o", "--segmentation-raster",
        type=Path,
        default=default_seg_raster,
        help="Output path for full segmentation raster (same grid as input).",
    )
    parser.add_argument(
        "--skip-create",
        action="store_true",
        help="Do nothing; exit 0 if segmentation raster exists, else exit with error.",
    )
    parser.add_argument("--scale", type=float, default=100.0)
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--block-size", type=int, default=2048)
    args = parser.parse_args()

    raster_path = resolve_path(args.raster, project_root)
    boundary_path = resolve_path(args.boundary, project_root)
    seg_raster_path = resolve_path(args.segmentation_raster, project_root)

    new_layers_dir = project_root / "data/processed/raster/synthetic_parenthesis"
    _move_layers_from_tiles_raster_if_needed(project_root, new_layers_dir)

    if args.skip_create:
        if seg_raster_path.exists():
            print(f"Segmentation raster exists: {seg_raster_path}")
            return
        raise FileNotFoundError(
            f"Segmentation raster not found: {seg_raster_path}. Run without --skip-create first."
        )

    if not raster_path.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")
    print(f"Using RGB for segments: {raster_path}")
    print("Creating segmentation layer (Felzenszwalb)...")
    script_dir = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        str(script_dir / "create_segmentation_layer.py"),
        "-i", str(raster_path),
        "-o", str(seg_raster_path),
        "-b", str(boundary_path),
        "--scale", str(args.scale),
        "--sigma", str(args.sigma),
        "--block-size", str(args.block_size),
    ]
    subprocess.run(cmd, check=True, cwd=project_root)
    print(f"Wrote {seg_raster_path}")


if __name__ == "__main__":
    main()
