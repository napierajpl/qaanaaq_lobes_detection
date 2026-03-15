#!/usr/bin/env python3
"""
Classify tiles as sun / shadow / mixed from a shadow-mask vector.

Polygons in the mask = shadow; outside polygons = sun. For each tile (geographic_bounds),
shadow fraction = (tile ∩ shadow_polygons area) / tile area. Classification:
  - shadow_fraction >= (1 - mixed_threshold) → shadow
  - shadow_fraction <= mixed_threshold → sun
  - else → mixed

Updates tile_registry.json and filtered_tiles.json (illumination, illumination_metrics={}).
Optionally writes a QGIS layer (GeoJSON + QML). Use this instead of add_illumination_tags
when you have a hand-drawn shadow mask.
"""

import json
import sys
from pathlib import Path

import yaml
from shapely.geometry import box
from shapely.ops import unary_union
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.path_utils import get_project_root, resolve_path


def load_registry(registry_path: Path) -> tuple[dict, dict]:
    with open(registry_path, encoding="utf-8") as f:
        data = json.load(f)
    return data, data.get("tiles", {})


def load_shadow_union(vector_path: Path, target_crs: str):
    import geopandas as gpd
    gdf = gpd.read_file(vector_path)
    if gdf.crs is None and target_crs:
        gdf.set_crs(target_crs, inplace=True)
    if target_crs and gdf.crs and str(gdf.crs) != str(target_crs):
        gdf = gdf.to_crs(target_crs)
    gdf = gdf[gdf.geometry.notnull()].copy()
    if gdf.empty:
        return None
    return unary_union(gdf.geometry.tolist())


def classify_tiles_by_shadow_mask(
    reg_tiles: dict,
    shadow_union,
    mixed_threshold: float,
) -> dict:
    # Returns tile_id -> "sun" | "shadow" | "mixed". No bounds or zero area → "mixed".
    result = {}
    for tid, entry in tqdm(reg_tiles.items(), desc="Classifying tiles", unit="tile"):
        bounds = entry.get("geographic_bounds", {})
        if not bounds:
            result[tid] = "mixed"
            continue
        minx, miny = bounds["minx"], bounds["miny"]
        maxx, maxy = bounds["maxx"], bounds["maxy"]
        tile_box = box(minx, miny, maxx, maxy)
        tile_area = tile_box.area
        if tile_area <= 0:
            result[tid] = "mixed"
            continue
        if shadow_union is None or shadow_union.is_empty:
            result[tid] = "sun"
            continue
        try:
            inter = tile_box.intersection(shadow_union)
            shadow_area = inter.area if inter else 0.0
        except Exception:
            result[tid] = "mixed"
            continue
        shadow_frac = shadow_area / tile_area
        if shadow_frac >= (1.0 - mixed_threshold):
            result[tid] = "shadow"
        elif shadow_frac <= mixed_threshold:
            result[tid] = "sun"
        else:
            result[tid] = "mixed"
    return result


def write_qgis_layer(registry_data: dict, reg_tiles: dict, output_path: Path) -> None:
    import geopandas as gpd
    from shapely.geometry import box

    rows = []
    for tid, entry in reg_tiles.items():
        if "illumination" not in entry:
            continue
        bounds = entry.get("geographic_bounds", {})
        if not bounds:
            continue
        geom = box(
            bounds["minx"], bounds["miny"],
            bounds["maxx"], bounds["maxy"],
        )
        rows.append({
            "tile_id": tid,
            "tile_label": tid,
            "illumination": entry["illumination"],
            "geometry": geom,
        })
    if not rows:
        return
    gdf = gpd.GeoDataFrame(rows, crs=registry_data.get("metadata", {}).get("crs") or registry_data.get("crs"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".shp":
        gdf_shp = gdf.rename(columns={"illumination": "illum"})
        gdf_shp.to_file(output_path, driver="ESRI Shapefile")
        _write_qml(output_path.with_suffix(".qml"), attr_name="illum")
    else:
        gdf.to_file(output_path, driver="GeoJSON")
        _write_qml(output_path.with_suffix(".qml"), attr_name="illumination")
    print(f"Wrote QGIS layer: {output_path} and {output_path.with_suffix('.qml')}")


def _write_qml(qml_path: Path, attr_name: str = "illumination") -> None:
    qml = f'''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28.0-Firenze" styleCategories="Symbology|Labeling" labelsEnabled="1">
  <renderer-v2 symbollevels="0" enableorderby="0" forceraster="0" type="categorizedSymbol" attr="{attr_name}">
    <categories>
      <category symbol="0" value="shadow" label="Shadow" render="true"/>
      <category symbol="1" value="sun" label="Sun" render="true"/>
      <category symbol="2" value="mixed" label="Mixed" render="true"/>
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
        description="Classify tiles by shadow-mask vector (polygons=shadow, outside=sun); mixed if >30% other class."
    )
    parser.add_argument(
        "--shadow-mask",
        type=Path,
        default=project_root / "data/raw/vector/shadow_mask.shp",
        help="Path to shadow mask shapefile (polygons = shadow)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs/data_preparation_config.yaml",
        help="Config for paths (registry, filtered_tiles)",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Override: tile_registry.json path",
    )
    parser.add_argument(
        "--filtered-tiles",
        type=Path,
        default=None,
        help="Override: filtered_tiles.json path",
    )
    parser.add_argument(
        "--mixed-threshold",
        type=float,
        default=None,
        help="Fraction threshold: tile is mixed if > this is the minority class (default from config or 0.3)",
    )
    parser.add_argument(
        "--qgis-layer",
        type=Path,
        default=None,
        help="Output QGIS GeoJSON path (default: same dir as registry, illumination_tiles.geojson)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts only; do not write JSON or QGIS layer",
    )
    args = parser.parse_args()

    with open(resolve_path(args.config, project_root), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    illum_config = config.get("illumination", {})
    tile_size = config.get("tile_size", 512)
    paths_key = "paths_512" if tile_size == 512 else "paths_256"
    paths = config.get(paths_key, {})
    filtered_path = args.filtered_tiles or resolve_path(
        Path(paths.get("filtered_tiles", "data/processed/tiles/train_512/filtered_tiles.json")),
        project_root,
    )
    registry_path = args.registry or filtered_path.parent / "tile_registry.json"
    qgis_layer_path = args.qgis_layer or filtered_path.parent / "illumination_tiles.geojson"

    if not registry_path.exists():
        print(f"Error: registry not found: {registry_path}")
        sys.exit(1)

    shadow_path_raw = args.shadow_mask or illum_config.get("shadow_mask")
    shadow_path = resolve_path(Path(shadow_path_raw), project_root) if shadow_path_raw else None
    if not shadow_path or not Path(shadow_path).exists():
        print(f"Error: shadow mask not found: {shadow_path}")
        sys.exit(1)

    registry_data, reg_tiles = load_registry(registry_path)
    registry_crs = registry_data.get("metadata", {}).get("crs") or registry_data.get("crs")

    print(f"Loading shadow mask: {shadow_path}")
    shadow_union = load_shadow_union(Path(shadow_path), registry_crs)

    mixed_threshold = float(
        args.mixed_threshold if args.mixed_threshold is not None else illum_config.get("mixed_threshold", 0.3)
    )
    classifications = classify_tiles_by_shadow_mask(reg_tiles, shadow_union, mixed_threshold)

    counts = {"shadow": 0, "sun": 0, "mixed": 0}
    for tid, tag in classifications.items():
        reg_tiles[tid]["illumination"] = tag
        reg_tiles[tid]["illumination_metrics"] = {}
        counts[tag] += 1

    print(f"Illumination (shadow mask): shadow={counts['shadow']}, sun={counts['sun']}, mixed={counts['mixed']}")

    filtered_data = None
    if filtered_path.exists():
        with open(filtered_path, encoding="utf-8") as f:
            filtered_data = json.load(f)
        tiles_list = filtered_data.get("tiles", [])
        for t in tiles_list:
            tid = t.get("tile_id")
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
        write_qgis_layer(registry_data, reg_tiles, qgis_layer_path)
    else:
        print("Dry run: no files written.")


if __name__ == "__main__":
    main()
