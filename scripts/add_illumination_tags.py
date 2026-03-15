#!/usr/bin/env python3
"""
Add illumination tags (shadow / sun / ambiguous) to all tiles inside the research boundary.

Two modes:
  - Vector mask (recommended): provide a hand-drawn polygon layer with an attribute (e.g. "illumination"
    = sun/shadow/ambiguous). Tiles are tagged by tile-centroid point-in-polygon. Set illumination_vector
    in config or --illumination-vector.
  - From RGB: load each tile's feature GeoTIFF, compute HSV, calibrate from example tile IDs; tag by
    mean brightness thresholds. Used when no illumination_vector is set.

Outputs: updates tile_registry.json and filtered_tiles.json; writes illumination_tiles.geojson + .qml for QGIS.
Config: configs/data_preparation_config.yaml (illumination section).
"""

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.path_utils import get_project_root, resolve_path


def normalize_tile_id(x) -> str:
    if isinstance(x, int):
        return f"tile_{x:04d}"
    s = str(x).strip()
    if s.isdigit():
        return f"tile_{int(s):04d}"
    if s.startswith("tile_"):
        return s
    return f"tile_{s}"


def rgb_to_hsv_per_tile(rgb: np.ndarray) -> tuple:
    """rgb (3, H, W) 0-255. Returns (mean_hue, mean_sat, mean_value) in [0,1]."""
    from matplotlib.colors import rgb_to_hsv

    r = np.clip(rgb[0].astype(np.float64) / 255.0, 0, 1)
    g = np.clip(rgb[1].astype(np.float64) / 255.0, 0, 1)
    b = np.clip(rgb[2].astype(np.float64) / 255.0, 0, 1)
    rgb_flat = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=1)
    hsv = rgb_to_hsv(rgb_flat)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    mask = (v >= 0.05) & (v <= 0.95)
    if np.sum(mask) < 100:
        mask = np.ones_like(v, dtype=bool)
    return float(np.mean(h[mask])), float(np.mean(s[mask])), float(np.mean(v[mask]))


def load_registry(registry_path: Path) -> tuple[dict, dict]:
    with open(registry_path, encoding="utf-8") as f:
        data = json.load(f)
    return data, data.get("tiles", {})


def inside_boundary_tile_ids(reg_tiles: dict) -> set:
    return {tid for tid, entry in reg_tiles.items() if entry.get("inside_boundary") is True}


def load_illumination_from_vector(
    vector_path: Path,
    reg_tiles: dict,
    inside_ids: set,
    registry_crs: str | None,
    attribute_name: str = "illumination",
) -> dict:
    # Tile centroid point-in-polygon; returns tile_id -> "sun"|"shadow"|"ambiguous". No polygon → ambiguous.
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.read_file(vector_path)
    if gdf.crs is None and registry_crs:
        gdf.set_crs(registry_crs, inplace=True)
    if registry_crs and gdf.crs and str(gdf.crs) != str(registry_crs):
        gdf = gdf.to_crs(registry_crs)
    if attribute_name not in gdf.columns:
        raise ValueError(f"Vector has no attribute '{attribute_name}'. Columns: {list(gdf.columns)}")
    gdf = gdf[gdf.geometry.notnull()].copy()
    if gdf.empty:
        raise ValueError("Vector has no valid geometries")
    valid = {"sun", "shadow", "ambiguous"}
    result = {}
    for tid in inside_ids:
        entry = reg_tiles.get(tid, {})
        bounds = entry.get("geographic_bounds", {})
        if not bounds:
            result[tid] = "ambiguous"
            continue
        cx = (bounds["minx"] + bounds["maxx"]) / 2.0
        cy = (bounds["miny"] + bounds["maxy"]) / 2.0
        pt = Point(cx, cy)
        found = None
        for _, row in gdf.iterrows():
            if row.geometry and row.geometry.contains(pt):
                val = str(row[attribute_name]).strip().lower() if row[attribute_name] is not None else ""
                if val in valid:
                    found = val
                else:
                    found = "ambiguous"
                break
        result[tid] = found if found is not None else "ambiguous"
    return result


def resolve_feature_path(
    tile_id: str,
    features_dir: Path,
    filtered_by_id: dict,
    reg_tiles: dict,
) -> Path | None:
    def try_path(base: Path, rel: str) -> Path | None:
        if not rel:
            return None
        p = base / rel
        if p.exists():
            return p
        p = next(base.glob(f"**/{Path(rel).name}"), None)
        return p

    if tile_id in filtered_by_id:
        rel = filtered_by_id[tile_id].get("features_path", "").replace("\\", "/")
        p = try_path(features_dir, rel)
        if p is not None:
            return p
        p = try_path(features_dir.parent, rel)
        if p is not None:
            return p
    entry = reg_tiles.get(tile_id, {})
    paths = entry.get("paths", {})
    feat_rel = paths.get("features", "").replace("\\", "/")
    if feat_rel:
        p = try_path(features_dir, feat_rel)
        if p is not None:
            return p
        p = try_path(features_dir.parent, feat_rel)
        if p is not None:
            return p
    candidate = next(features_dir.glob(f"**/{tile_id}.tif"), None)
    if candidate is not None:
        return candidate
    return next(features_dir.parent.glob(f"**/{tile_id}.tif"), None)


def compute_tile_hsv(features_path: Path, _log_first_error: list | None = None) -> tuple | None:
    if not features_path or not features_path.exists():
        return None
    try:
        with rasterio.open(features_path) as src:
            if src.count < 3:
                return None
            rgb = src.read([1, 2, 3])
        return rgb_to_hsv_per_tile(rgb)
    except Exception as e:
        if _log_first_error is not None and len(_log_first_error) == 0:
            _log_first_error.append(f"{features_path}: {e}")
        return None


def calibrate_thresholds(
    tile_metrics: dict,
    shadow_ids: list,
    sun_ids: list,
    ambiguous_band: float,
    ambiguous_max_fraction: float | None = None,
) -> tuple:
    all_vals = [tile_metrics[tid][2] for tid in tile_metrics]
    if not all_vals:
        return 0.4, 0.6
    if ambiguous_max_fraction is not None and 0 < ambiguous_max_fraction < 1:
        half = (1 - ambiguous_max_fraction) / 2.0
        low_thresh = float(np.percentile(all_vals, 100 * half))
        high_thresh = float(np.percentile(all_vals, 100 * (1 - half)))
        return low_thresh, high_thresh
    shadow_ids = [normalize_tile_id(i) for i in shadow_ids]
    sun_ids = [normalize_tile_id(i) for i in sun_ids]
    shadow_vals = [tile_metrics[tid][2] for tid in shadow_ids if tid in tile_metrics]
    sun_vals = [tile_metrics[tid][2] for tid in sun_ids if tid in tile_metrics]
    if not shadow_vals or not sun_vals:
        mid = 0.45
        return mid - ambiguous_band / 2, mid + ambiguous_band / 2
    low_mean = float(np.mean(shadow_vals))
    high_mean = float(np.mean(sun_vals))
    mid = (low_mean + high_mean) / 2.0
    half = ambiguous_band / 2.0
    return mid - half, mid + half


def classify_illumination(mean_value: float, low_thresh: float, high_thresh: float) -> str:
    if mean_value < low_thresh:
        return "shadow"
    if mean_value > high_thresh:
        return "sun"
    return "ambiguous"


def write_illumination_qgis_layer(
    registry_data: dict,
    reg_tiles: dict,
    output_path: Path,
) -> None:
    import geopandas as gpd
    from shapely.geometry import box

    crs_str = registry_data.get("metadata", {}).get("crs")
    geometries = []
    rows = []
    for tile_id, entry in reg_tiles.items():
        illum = entry.get("illumination")
        if illum is None:
            continue
        bounds = entry.get("geographic_bounds", {})
        if not bounds:
            continue
        geom = box(
            bounds["minx"],
            bounds["miny"],
            bounds["maxx"],
            bounds["maxy"],
        )
        geometries.append(geom)
        metrics = entry.get("illumination_metrics", {})
        rows.append({
            "tile_id": tile_id,
            "tile_label": tile_id.replace("tile_", "") if tile_id.startswith("tile_") else tile_id,
            "illumination": illum,
            "mean_value": metrics.get("mean_value"),
            "mean_hue": metrics.get("mean_hue"),
            "mean_sat": metrics.get("mean_saturation"),
        })
    if not rows:
        return
    gdf = gpd.GeoDataFrame(rows, geometry=geometries)
    if crs_str:
        try:
            gdf.set_crs(crs_str, inplace=True)
        except (ValueError, Exception):
            pass
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower()
    if ext == ".shp":
        gdf.to_file(output_path, driver="ESRI Shapefile")
    else:
        gdf.to_file(output_path, driver="GeoJSON")
    qml_path = output_path.with_suffix(".qml")
    _write_illumination_qml(qml_path)
    print(f"Wrote QGIS layer: {output_path} and style {qml_path}")


def _write_illumination_qml(qml_path: Path) -> None:
    qml = '''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28.0-Firenze" styleCategories="Symbology|Labeling" labelsEnabled="1">
  <renderer-v2 symbollevels="0" enableorderby="0" forceraster="0" type="categorizedSymbol" attr="illumination">
    <categories>
      <category symbol="0" value="shadow" label="Shadow" render="true"/>
      <category symbol="1" value="sun" label="Sun" render="true"/>
      <category symbol="2" value="ambiguous" label="Ambiguous" render="true"/>
    </categories>
    <symbols>
      <symbol alpha="0.5" clip_to_extent="1" force_rhr="0" name="0" type="fill">
        <layer locked="0" class="SimpleFill" pass="0" enabled="1">
          <Option type="Map">
            <Option name="color" value="60,60,80,255" type="QString"/>
            <Option name="outline_color" value="40,40,60,255" type="QString"/>
            <Option name="outline_style" value="solid" type="QString"/>
            <Option name="outline_width" value="0.26" type="QString"/>
            <Option name="style" value="solid" type="QString"/>
          </Option>
        </layer>
      </symbol>
      <symbol alpha="0.5" clip_to_extent="1" force_rhr="0" name="1" type="fill">
        <layer locked="0" class="SimpleFill" pass="0" enabled="1">
          <Option type="Map">
            <Option name="color" value="255,220,100,255" type="QString"/>
            <Option name="outline_color" value="200,170,50,255" type="QString"/>
            <Option name="outline_style" value="solid" type="QString"/>
            <Option name="outline_width" value="0.26" type="QString"/>
            <Option name="style" value="solid" type="QString"/>
          </Option>
        </layer>
      </symbol>
      <symbol alpha="0.45" clip_to_extent="1" force_rhr="0" name="2" type="fill">
        <layer locked="0" class="SimpleFill" pass="0" enabled="1">
          <Option type="Map">
            <Option name="color" value="150,150,150,255" type="QString"/>
            <Option name="outline_color" value="100,100,100,255" type="QString"/>
            <Option name="outline_style" value="solid" type="QString"/>
            <Option name="outline_width" value="0.26" type="QString"/>
            <Option name="style" value="solid" type="QString"/>
          </Option>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <labeling type="simple">
    <settings calloutType="simple">
      <text-style fontItalic="0" fontKerning="1" fontLetterSpacing="0" fontStrikeout="0" fontUnderline="0" fontWordSpacing="0" fieldName="tile_label" fontSize="8" fontSizeUnit="Point" blendMode="0" textColor="50,50,50,255" textOpacity="1" fontWeight="50" multilineHeight="1" useSubstitutions="0" fontSizeMapUnitScale="3x:0,0,0,0,0,0" previewBkgrdColor="255,255,255,255" fontFamily="Arial" namedStyle="Regular" allowHtml="0" isExpression="0">
        <text-buffer bufferSize="1" bufferSizeUnits="MM" bufferColor="255,255,255,255" bufferBlendMode="0" bufferDraw="1" bufferNoFill="0" bufferSizeMapUnitScale="3x:0,0,0,0,0,0" bufferJoinStyle="128" bufferOpacity="1"/>
      </text-style>
      <placement placementFlags="10" repeatDistance="0" dist="0" distUnits="MM" geometryGenerator="make_point(x_min($geometry), y_max($geometry))" geometryGeneratorType="PointGeometry" predefinedPositionOrder="TR,TL,BR,BL,R,L,TSR,BSR" offsetType="1" xOffset="5" yOffset="-5" lineAnchorType="0" rotationAngle="0" repeatDistanceUnits="MM" overrunDistance="0" priority="5" geometryGeneratorEnabled="1" maxCurvedCharAngleIn="25" maxCurvedCharAngleOut="-25" overrunDistanceUnits="MM" centroidWhole="0" labelOffsetMapUnitScale="3x:0,0,0,0,0,0" distMapUnitScale="3x:0,0,0,0,0,0" quadOffset="4" preserveRotation="1" layerType="PolygonGeometry" fitInPolygonOnly="0" placement="0"/>
      <rendering scaleVisibility="0" fontMinPixelSize="3" obstacle="1" upsidedownLabels="0" maxNumLabels="2000" zIndex="0" fontMaxPixelSize="10000" unplacedVisibility="0" mergeLines="0" minFeatureSize="0" limitNumLabels="0" drawLabels="1" scaleMin="0" scaleMax="0" obstacleType="1" labelPerPart="0" displayAll="0" obstacleFactor="1"/>
    </settings>
  </labeling>
  <blendMode>0</blendMode>
  <featureBlendMode>0</featureBlendMode>
  <layerGeometryType>2</layerGeometryType>
</qgis>
'''
    qml_path.parent.mkdir(parents=True, exist_ok=True)
    qml_path.write_text(qml, encoding="utf-8")


def main():
    import argparse

    project_root = get_project_root(Path(__file__))
    parser = argparse.ArgumentParser(
        description="Add illumination tags to all tiles inside boundary; update registry, filtered_tiles, and write QGIS layer."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs/data_preparation_config.yaml",
        help="Path to data preparation config (illumination section, paths)",
    )
    parser.add_argument(
        "--filtered-tiles",
        type=Path,
        default=None,
        help="Override: filtered_tiles.json path",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Override: tile_registry.json path",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=None,
        help="Override: features base dir for loading RGB",
    )
    parser.add_argument(
        "--qgis-layer",
        type=Path,
        default=None,
        help="Output QGIS layer path (default: same dir as registry, illumination_tiles.geojson)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print stats only; do not write JSON or QGIS layer",
    )
    parser.add_argument(
        "--illumination-vector",
        type=Path,
        default=None,
        help="Use hand-drawn vector (shapefile/GeoJSON) for illumination; tile centroid point-in-polygon. Overrides config.",
    )
    parser.add_argument(
        "--illumination-attribute",
        type=str,
        default=None,
        help="Attribute name in vector for sun/shadow/ambiguous (default from config or 'illumination')",
    )
    args = parser.parse_args()

    with open(resolve_path(args.config, project_root), encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tile_size = config.get("tile_size", 512)
    paths_key = "paths_512" if tile_size == 512 else "paths_256"
    paths = config.get(paths_key, {})
    illum_config = config.get("illumination", {})

    filtered_path = args.filtered_tiles or resolve_path(
        Path(paths.get("filtered_tiles", "data/processed/tiles/train_512/filtered_tiles.json")),
        project_root,
    )
    registry_path = args.registry or filtered_path.parent / "tile_registry.json"
    features_dir = args.features_dir or resolve_path(
        Path(paths.get("features_dir", "data/processed/tiles/train_512/features")),
        project_root,
    )
    qgis_layer_path = args.qgis_layer or filtered_path.parent / "illumination_tiles.geojson"

    if not registry_path.exists():
        print(f"Error: tile_registry not found: {registry_path}")
        sys.exit(1)

    registry_data, reg_tiles = load_registry(registry_path)
    inside_ids = inside_boundary_tile_ids(reg_tiles)
    print(f"Tiles inside boundary: {len(inside_ids)}")

    filtered_by_id = {}
    if filtered_path.exists():
        with open(filtered_path, encoding="utf-8") as f:
            filtered_data = json.load(f)
        tiles_list = filtered_data.get("tiles", [])
        filtered_by_id = {t["tile_id"]: t for t in tiles_list}
        print(f"Filtered tiles (for JSON update): {len(filtered_by_id)}")
    else:
        filtered_data = None
        print("filtered_tiles.json not found; only registry and QGIS layer will be updated.")

    vector_path_raw = args.illumination_vector or illum_config.get("illumination_vector")
    illumination_vector = resolve_path(Path(vector_path_raw), project_root) if vector_path_raw else None
    illumination_attribute = args.illumination_attribute or illum_config.get("illumination_attribute", "illumination")
    registry_crs = (registry_data.get("crs") or registry_data.get("metadata", {}).get("crs")) if registry_data else None

    if illumination_vector is not None:
        if not Path(illumination_vector).exists():
            print(f"Error: illumination vector not found: {illumination_vector}")
            sys.exit(1)
        print(f"Using illumination from vector: {illumination_vector} (attribute: {illumination_attribute})")
        vector_illum = load_illumination_from_vector(
            Path(illumination_vector), reg_tiles, inside_ids, registry_crs, illumination_attribute
        )
        counts = {"shadow": 0, "sun": 0, "ambiguous": 0}
        for tid in inside_ids:
            tag = vector_illum.get(tid, "ambiguous")
            reg_tiles[tid]["illumination"] = tag
            reg_tiles[tid]["illumination_metrics"] = {}
            counts[tag] += 1
        print(f"Illumination (vector): shadow={counts['shadow']}, sun={counts['sun']}, ambiguous={counts['ambiguous']}")
    else:
        shadow_ids = illum_config.get("shadow_example_ids", [])
        sun_ids = illum_config.get("sun_example_ids", [])
        ambiguous_band = float(illum_config.get("ambiguous_value_band", 0.08))
        ambiguous_max_fraction = illum_config.get("ambiguous_max_fraction")
        if ambiguous_max_fraction is not None:
            ambiguous_max_fraction = float(ambiguous_max_fraction)

        to_tag_ids = sorted(inside_ids)
        tile_metrics = {}
        first_error = []
        for tid in tqdm(to_tag_ids, desc="Computing illumination (all inside boundary)", unit="tile"):
            feat_path = resolve_feature_path(tid, features_dir, filtered_by_id, reg_tiles)
            hsv = compute_tile_hsv(Path(feat_path) if feat_path else None, _log_first_error=first_error)
            if hsv is not None:
                tile_metrics[tid] = hsv

        n_with_rgb = len(tile_metrics)
        n_no_rgb = len(to_tag_ids) - n_with_rgb
        if n_no_rgb:
            print(f"Tiles without readable RGB (skipped): {n_no_rgb}")
        if first_error:
            print(f"First open error (for debugging): {first_error[0]}")

        low_thresh, high_thresh = calibrate_thresholds(
            tile_metrics, shadow_ids, sun_ids, ambiguous_band,
            ambiguous_max_fraction=ambiguous_max_fraction,
        )
        if ambiguous_max_fraction is not None:
            print(f"Calibrated value thresholds: low={low_thresh:.3f}, high={high_thresh:.3f} (ambiguous max fraction={ambiguous_max_fraction})")
        else:
            print(f"Calibrated value thresholds: low={low_thresh:.3f}, high={high_thresh:.3f} (ambiguous band={ambiguous_band})")

        counts = {"shadow": 0, "sun": 0, "ambiguous": 0}
        for tid in to_tag_ids:
            if tid not in tile_metrics:
                continue
            _, _, v = tile_metrics[tid]
            tag = classify_illumination(v, low_thresh, high_thresh)
            metrics = {
                "mean_hue": tile_metrics[tid][0],
                "mean_saturation": tile_metrics[tid][1],
                "mean_value": tile_metrics[tid][2],
            }
            reg_tiles[tid]["illumination"] = tag
            reg_tiles[tid]["illumination_metrics"] = metrics
            counts[tag] += 1

        print(f"Illumination (inside boundary, with RGB): shadow={counts['shadow']}, sun={counts['sun']}, ambiguous={counts['ambiguous']}")

    if filtered_data is not None and "tiles" in filtered_data:
        for t in filtered_data["tiles"]:
            tid = t["tile_id"]
            if tid in reg_tiles and "illumination" in reg_tiles[tid]:
                t["illumination"] = reg_tiles[tid]["illumination"]
                t["illumination_metrics"] = reg_tiles[tid].get("illumination_metrics", {})
        if "stats" not in filtered_data:
            filtered_data["stats"] = {}
        filtered_data["stats"]["illumination_counts"] = counts

    if not args.dry_run:
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry_data, f, indent=2)
        print(f"Wrote {registry_path}")

        if filtered_data is not None:
            with open(filtered_path, "w", encoding="utf-8") as f:
                json.dump(filtered_data, f, indent=2)
            print(f"Wrote {filtered_path}")

        write_illumination_qgis_layer(registry_data, reg_tiles, qgis_layer_path)
    else:
        print("Dry run: no files written.")


if __name__ == "__main__":
    main()
