"""
Generate synthetic parenthesis dataset from full raster: embed shapes inside
boundary polygons (avoiding white), then tile to 256 and 512.
Produces same target raster for learning (0/20). Run extract_imagery_boundaries.py first.
"""
import json
from pathlib import Path

import numpy as np
import rasterio

from src.data_processing.boundary_extraction import (
    build_valid_mask,
    rasterize_boundaries_to_mask,
)
from src.data_processing.synthetic_shapes import (
    make_parenthesis_mask,
    place_one_shape,
    rotate_mask,
)
from src.data_processing.boundary_tile_filter import filter_filtered_tiles_by_boundary
from src.data_processing.tiling import Tiler
from src.map_overlays.tile_registry import TileRegistry
from src.utils.path_utils import get_project_root, resolve_path


def _filter_tiles_by_target_coverage(
    tiles: list[dict],
    targets_dir: Path,
    min_target_coverage: float,
    lobe_threshold: float = 0.5,
) -> list[dict]:
    """Keep only tiles where target has at least min_target_coverage fraction of positive pixels (e.g. lobe)."""
    if min_target_coverage <= 0:
        return tiles
    kept = []
    for t in tiles:
        tid = t.get("tile_id", "")
        tgt_path = targets_dir / f"{tid}.tif"
        if not tgt_path.exists():
            continue
        with rasterio.open(tgt_path) as src:
            data = src.read(1)
        total = data.size
        positive = int((data > lobe_threshold).sum())
        coverage = positive / total if total else 0
        if coverage >= min_target_coverage:
            kept.append(t)
    return kept

def _placement_candidates(
    boundary_mask: np.ndarray,
    rgb: np.ndarray,
    shape_height_px: int,
    white_threshold: int = 250,
) -> list[tuple[int, int]]:
    """Return list of (row, col) valid for shape center (inside boundary, margin for shape size)."""
    H, W = boundary_mask.shape
    margin = (shape_height_px // 2) + 20
    if H <= 2 * margin or W <= 2 * margin:
        return []
    not_white = ~(
        (rgb[0] >= white_threshold)
        & (rgb[1] >= white_threshold)
        & (rgb[2] >= white_threshold)
    )
    row_ok = (np.arange(H) >= margin) & (np.arange(H) < H - margin)
    col_ok = (np.arange(W) >= margin) & (np.arange(W) < W - margin)
    valid = boundary_mask.astype(bool) & not_white & row_ok[:, None] & col_ok[None, :]
    rows, cols = np.where(valid)
    return list(zip(rows.tolist(), cols.tolist()))


def _draw_shapes_on_raster(
    rgb: np.ndarray,
    target: np.ndarray,
    boundary_mask: np.ndarray,
    n_shapes: int,
    shape_height_px: int,
    white_fraction_max: float,
    white_threshold: int,
    rng: np.random.Generator,
) -> None:
    """Draw n_shapes parenthesis on rgb and target, only inside boundary and not on white."""
    H, W = target.shape
    candidates = _placement_candidates(boundary_mask, rgb, shape_height_px, white_threshold)
    if not candidates:
        return
    chars = ["(", ")"]
    placed = 0
    max_attempts = n_shapes * 20
    for _ in range(max_attempts):
        if placed >= n_shapes:
            break
        idx = rng.integers(0, len(candidates))
        center_r, center_c = candidates[idx]
        char = chars[rng.integers(0, 2)]
        mask = make_parenthesis_mask(char, shape_height_px)
        angle = rng.uniform(0, 360)
        mask = rotate_mask(mask, angle)
        mh, mw = mask.shape
        top = center_r - mh // 2
        left = center_c - mw // 2
        if top < 0 or left < 0 or top + mh > H or left + mw > W:
            continue
        roi_r = slice(top, top + mh)
        roi_c = slice(left, left + mw)
        patch_rgb = rgb[:, roi_r, roi_c]
        white = np.all(patch_rgb >= white_threshold, axis=0)
        under_shape = mask > 0
        if np.mean(white[under_shape]) > white_fraction_max:
            continue
        place_one_shape(rgb, target, center_r, center_c, mask, target_value=20)
        placed += 1


def _write_raster_from_array(
    path: Path,
    data: np.ndarray,
    profile: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = profile.copy()
    profile.update(
        count=data.shape[0],
        height=data.shape[1],
        width=data.shape[2],
        dtype=data.dtype,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def _stack_rgb_dem_slope(
    rgb_path: Path,
    dem_path: Path,
    slope_path: Path,
    output_path: Path,
) -> None:
    """Write 5-band GeoTIFF: RGB from rgb_path, then DEM, then Slope (same grid)."""
    with rasterio.open(rgb_path) as rgb_src:
        profile = rgb_src.profile.copy()
        H, W = rgb_src.height, rgb_src.width
        rgb = rgb_src.read()
    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(out_shape=(H, W), resampling=rasterio.enums.Resampling.bilinear)
    with rasterio.open(slope_path) as slope_src:
        slope = slope_src.read(out_shape=(H, W), resampling=rasterio.enums.Resampling.bilinear)
    stack = np.concatenate([rgb, dem, slope], axis=0).astype(np.float32)
    profile.update(count=5, dtype=np.float32)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(stack)


def generate_from_raster(
    raster_path: Path,
    boundaries_path: Path,
    dem_path: Path,
    slope_path: Path,
    output_base: Path,
    tile_sizes: list[int],
    raster_output_dir: Path | None = None,
    n_shapes: int = 2000,
    shape_height_px: int = 300,
    white_threshold: int = 250,
    white_fraction_max: float = 0.8,
    overlap: float = 0.3,
    seed: int = 42,
    min_target_coverage: float = 0.0,
    filter_by_boundary: bool = False,
) -> None:
    """Load full raster, place shapes inside boundaries, write full layers to raster_output_dir, then tile from there."""
    rng = np.random.default_rng(seed)

    with rasterio.open(raster_path) as src:
        rgb = src.read([1, 2, 3])
        profile = src.profile.copy()
        profile.update(count=3)
        H, W = src.height, src.width

    if boundaries_path.exists():
        try:
            import geopandas as gpd
            gdf = gpd.read_file(boundaries_path)
            if gdf.empty:
                print("Boundary vector is empty; using valid-data mask (non-white) for placement.")
                boundary_mask = build_valid_mask(rgb, white_threshold=white_threshold)
            else:
                boundary_mask = rasterize_boundaries_to_mask(boundaries_path, raster_path)
                if boundary_mask.shape[0] != H or boundary_mask.shape[1] != W:
                    boundary_mask = np.ones((H, W), dtype=np.uint8)
        except Exception as e:
            print(f"Could not use boundary vector ({e}); using valid-data mask.")
            boundary_mask = build_valid_mask(rgb, white_threshold=white_threshold)
    else:
        boundary_mask = build_valid_mask(rgb, white_threshold=white_threshold)
    boundary_mask = boundary_mask.astype(np.uint8)

    target = np.zeros((H, W), dtype=np.float32)
    _draw_shapes_on_raster(
        rgb, target, boundary_mask,
        n_shapes=n_shapes,
        shape_height_px=shape_height_px,
        white_fraction_max=white_fraction_max,
        white_threshold=white_threshold,
        rng=rng,
    )

    if raster_output_dir is None:
        raster_output_dir = output_base / "raster"
    raster_output_dir = Path(raster_output_dir)
    raster_output_dir.mkdir(parents=True, exist_ok=True)
    rgb_out = raster_output_dir / "synthetic_rgb_with_shapes.tif"
    target_out = raster_output_dir / "synthetic_target.tif"
    features_5band = raster_output_dir / "synthetic_features_5band.tif"

    _write_raster_from_array(rgb_out, rgb, profile)
    profile_t = profile.copy()
    profile_t.update(count=1, dtype=np.float32)
    _write_raster_from_array(target_out, target.reshape(1, H, W), profile_t)

    _stack_rgb_dem_slope(rgb_out, dem_path, slope_path, features_5band)

    for tile_size in tile_sizes:
        out_dir = output_base / f"synthetic_parenthesis_{tile_size}"
        feat_dir = out_dir / "features"
        tgt_dir = out_dir / "targets"
        feat_dir.mkdir(parents=True, exist_ok=True)
        tgt_dir.mkdir(parents=True, exist_ok=True)

        tiler = Tiler(tile_size=tile_size, overlap=overlap)
        feature_paths = tiler.tile_raster(
            features_5band, feat_dir, base_filename="features", organize_by_source=False
        )
        target_paths = tiler.tile_raster(
            target_out, tgt_dir, base_filename="targets", organize_by_source=False
        )

        seg_raster = raster_output_dir / "segmentation_layer.tif"
        if seg_raster.exists():
            seg_dir = out_dir / "segmentation"
            seg_dir.mkdir(parents=True, exist_ok=True)
            tiler.tile_raster(
                seg_raster, seg_dir, base_filename="segmentation", organize_by_source=False
            )
            print(f"  Tiled segmentation to {seg_dir.relative_to(output_base)}")

        tile_ids = [p.stem for p in feature_paths]
        tiles = [
            {
                "tile_id": tid,
                "features_path": f"{tid}.tif",
                "targets_path": f"{tid}.tif",
            }
            for tid in tile_ids
        ]
        n_before = len(tiles)
        tiles = _filter_tiles_by_target_coverage(
            tiles, tgt_dir, min_target_coverage, lobe_threshold=0.5
        )
        if n_before > len(tiles):
            print(f"  Filtered to {len(tiles)} tiles with target coverage >= {min_target_coverage} (dropped {n_before - len(tiles)} empty/low-content)")
        filtered_path = out_dir / "filtered_tiles.json"
        with open(filtered_path, "w", encoding="utf-8") as f:
            json.dump({"tiles": tiles, "tile_size": tile_size}, f, indent=2)
        if filter_by_boundary and boundaries_path.exists() and tiles:
            n_after = filter_filtered_tiles_by_boundary(
                filtered_path, boundaries_path, filtered_path, features_dir=feat_dir
            )
            print(f"  Filtered by boundary to {n_after} tiles inside {boundaries_path.name}")
            with open(filtered_path, encoding="utf-8") as f:
                tiles = json.load(f)["tiles"]
        print(f"Wrote {len(tiles)} tiles to {out_dir} (tile_size={tile_size})")

        if features_5band.exists():
            registry_path = out_dir / "tile_registry.json"
            registry = TileRegistry(registry_path, features_5band)
            registry.migrate_from_filtered_tiles(
                filtered_tiles_path=filtered_path,
                source_raster_path=features_5band,
                features_dir=feat_dir,
                train_split=0.6,
                val_split=0.2,
                test_split=0.2,
                tile_size=tile_size,
                overlap=overlap,
            )
            registry.add_boundary_info(boundaries_path)
            registry.save()
            print(f"  Wrote {registry_path.name} (with inside_boundary)")


def main() -> None:
    project_root = get_project_root(Path(__file__))
    default_raster = project_root / "data/raw/raster/imagery/qaanaaq_rgb_0_2m.tif"
    default_boundaries = project_root / "data/raw/vector/research_boundary.shp"
    default_dem = project_root / "data/processed/raster/dem_from_arcticDEM_resampled.tif"
    default_slope = project_root / "data/processed/raster/slope_from_dem_resampled.tif"
    default_out = project_root / "data/processed/tiles"
    default_raster_out = project_root / "data/processed/raster/synthetic_parenthesis"

    import argparse
    parser = argparse.ArgumentParser(
        description="Generate synthetic parenthesis from full raster (inside boundaries), then tile.",
    )
    parser.add_argument("-i", "--raster", type=Path, default=default_raster)
    parser.add_argument(
        "-b", "--boundaries",
        type=Path,
        default=default_boundaries,
        help="Boundary vector (shapefile/GeoJSON); parenthesis placed inside. Default: data/raw/vector/research_boundary.shp",
    )
    parser.add_argument("--dem", type=Path, default=default_dem)
    parser.add_argument("--slope", type=Path, default=default_slope)
    parser.add_argument("-o", "--output-base", type=Path, default=default_out)
    parser.add_argument(
        "--raster-output",
        type=Path,
        default=default_raster_out,
        help="Directory for full layers (RGB with shapes, target, 5-band). Default: data/processed/raster/synthetic_parenthesis",
    )
    parser.add_argument("--tile-sizes", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--n-shapes", type=int, default=2000)
    parser.add_argument("--shape-height-px", type=int, default=300)
    parser.add_argument("--white-threshold", type=int, default=250)
    parser.add_argument("--white-fraction-max", type=float, default=0.8)
    parser.add_argument("--overlap", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-target-coverage",
        type=float,
        default=0.0,
        help="Exclude tiles with target lobe coverage below this (0 = keep all). E.g. 0.0001 to drop empty tiles.",
    )
    parser.add_argument(
        "--filter-by-boundary",
        action="store_true",
        help="Keep only tiles that intersect the boundary (same as -b). Reduces training to AOI.",
    )
    args = parser.parse_args()

    raster_path = resolve_path(args.raster, project_root)
    boundaries_path = resolve_path(args.boundaries, project_root)
    dem_path = resolve_path(args.dem, project_root)
    slope_path = resolve_path(args.slope, project_root)
    output_base = resolve_path(args.output_base, project_root)
    raster_output_dir = resolve_path(args.raster_output, project_root)

    if not raster_path.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")
    if not slope_path.exists():
        raise FileNotFoundError(f"Slope not found: {slope_path}")

    generate_from_raster(
        raster_path=raster_path,
        boundaries_path=boundaries_path,
        dem_path=dem_path,
        slope_path=slope_path,
        output_base=output_base,
        tile_sizes=args.tile_sizes,
        raster_output_dir=raster_output_dir,
        n_shapes=args.n_shapes,
        shape_height_px=args.shape_height_px,
        white_threshold=args.white_threshold,
        white_fraction_max=args.white_fraction_max,
        overlap=args.overlap,
        seed=args.seed,
        min_target_coverage=args.min_target_coverage,
        filter_by_boundary=args.filter_by_boundary,
    )


if __name__ == "__main__":
    main()
