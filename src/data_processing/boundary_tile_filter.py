"""
Filter tiles to those intersecting a boundary (e.g. research boundary).
Used to limit training and other processes to the area of interest.
"""
from pathlib import Path
from typing import List, Set

import geopandas as gpd
from shapely.geometry import box


def load_boundary_union(boundary_path: Path):
    """Load vector and return unary_union of geometries (for intersection tests)."""
    gdf = gpd.read_file(boundary_path)
    if gdf.empty:
        return None
    return gdf.unary_union


def tile_bounds_intersect_boundary(
    geographic_bounds: dict,
    boundary_union,
) -> bool:
    """Return True if tile bounds intersect the boundary."""
    if boundary_union is None:
        return True
    minx = geographic_bounds.get("minx")
    miny = geographic_bounds.get("miny")
    maxx = geographic_bounds.get("maxx")
    maxy = geographic_bounds.get("maxy")
    if None in (minx, miny, maxx, maxy):
        return True
    tile_box = box(minx, miny, maxx, maxy)
    return tile_box.intersects(boundary_union)


def tile_ids_inside_boundary_from_registry(
    registry_path: Path,
    boundary_path: Path,
) -> Set[str]:
    """Return set of tile_ids that have geographic_bounds intersecting the boundary."""
    import json

    boundary_union = load_boundary_union(Path(boundary_path))
    with open(registry_path, encoding="utf-8") as f:
        registry = json.load(f)
    tiles = registry.get("tiles", {})
    inside = set()
    for tile_id, entry in tiles.items():
        bounds = entry.get("geographic_bounds")
        if not bounds:
            inside.add(tile_id)
            continue
        if tile_bounds_intersect_boundary(bounds, boundary_union):
            inside.add(tile_id)
    return inside


def tile_ids_inside_boundary_from_feature_dir(
    filtered_tiles: List[dict],
    features_dir: Path,
    boundary_path: Path,
) -> Set[str]:
    """Get tile bounds from feature GeoTIFFs and return tile_ids intersecting boundary."""
    import rasterio

    boundary_union = load_boundary_union(Path(boundary_path))
    features_dir = Path(features_dir)
    inside = set()
    for tile in filtered_tiles:
        tile_id = tile.get("tile_id", "")
        feat_rel = tile.get("features_path", "").replace("\\", "/")
        feat_path = features_dir / feat_rel if not Path(feat_rel).is_absolute() else Path(feat_rel)
        if not feat_path.exists():
            continue
        with rasterio.open(feat_path) as src:
            b = src.bounds
        geographic_bounds = {"minx": b.left, "miny": b.bottom, "maxx": b.right, "maxy": b.top}
        if tile_bounds_intersect_boundary(geographic_bounds, boundary_union):
            inside.add(tile_id)
    return inside


def filter_filtered_tiles_by_boundary(
    filtered_tiles_path: Path,
    boundary_path: Path,
    output_path: Path,
    features_dir: Path | None = None,
    registry_path: Path | None = None,
) -> int:
    """
    Write a new filtered_tiles.json containing only tiles that intersect the boundary.
    Uses registry for bounds if provided; otherwise reads each feature tile for bounds.
    Returns number of tiles written.
    """
    import json

    with open(filtered_tiles_path, encoding="utf-8") as f:
        data = json.load(f)
    tiles = data.get("tiles", [])
    if registry_path and Path(registry_path).exists():
        inside = tile_ids_inside_boundary_from_registry(registry_path, boundary_path)
    elif features_dir and Path(features_dir).exists():
        inside = tile_ids_inside_boundary_from_feature_dir(tiles, features_dir, boundary_path)
    else:
        raise ValueError("Provide either --registry or --features-dir to get tile bounds")

    filtered = [t for t in tiles if t.get("tile_id") in inside]
    out_data = {k: v for k, v in data.items() if k != "tiles"}
    out_data["tiles"] = filtered
    if "tile_size" in data:
        out_data["tile_size"] = data["tile_size"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    return len(filtered)
